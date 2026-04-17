# yolo_email_streamlit.py
# =========================================================
# Streamlit 終極版 v3：YOLO 偵測 + 智慧存檔 + Email 警報 (非同步發送)
# =========================================================

from __future__ import annotations

import os
import ncnn
import time
import threading  # 🔥 新增：用於背景發送信件
import ssl
import smtplib
from email.message import EmailMessage
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import cv2  # type: ignore
import numpy as np
import streamlit as st
from PIL import Image

# 避免 OpenMP 錯誤
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =====================================================
# 📧 0. Email 發送工具 (您提供的程式碼)
# =====================================================
def send_email(
    smtp_host: str,
    smtp_port: int,
    username: str,
    password: str,
    sender: str,
    to: list[str],
    subject: str,
    body_text: Optional[str] = None,
    html_body: Optional[str] = None,
    cc: Optional[list[str]] = None,
    bcc: Optional[list[str]] = None,
    attachments: Optional[Iterable[str]] = None,
    use_ssl: bool = True,
    timeout: float = 30.0,
) -> None:
    """
    通用寄信函式（支援純文字/HTML/附件，SSL 或 STARTTLS）。
    """
    if not body_text and not html_body:
        raise ValueError("至少需要 body_text 或 html_body 其中一個。")

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = ", ".join(to)
    if cc:
        msg["Cc"] = ", ".join(cc)
    msg["Subject"] = subject

    if body_text and html_body:
        msg.set_content(body_text)
        msg.add_alternative(html_body, subtype="html")
    elif html_body:
        msg.set_content("Your email client does not support HTML.")
        msg.add_alternative(html_body, subtype="html")
    else:
        msg.set_content(body_text or "")

    # 附件處理
    if attachments:
        import mimetypes
        for path_str in attachments:
            path = Path(path_str)
            # 確保檔案存在
            if not path.exists():
                print(f"⚠️ 找不到附件: {path_str}")
                continue
                
            ctype, encoding = mimetypes.guess_type(path.name)
            if ctype is None or encoding is not None:
                ctype = "application/octet-stream"
            maintype, subtype = ctype.split("/", 1)
            with path.open("rb") as f:
                data = f.read()
            msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=path.name)

    recipients = list(to) + (cc or []) + (bcc or [])

    try:
        if use_ssl:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(host=smtp_host, port=smtp_port, context=context, timeout=timeout) as server:
                server.login(username, password)
                server.send_message(msg, from_addr=sender, to_addrs=recipients)
        else:
            with smtplib.SMTP(host=smtp_host, port=smtp_port, timeout=timeout) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(username, password)
                server.send_message(msg, from_addr=sender, to_addrs=recipients)
        print("✅ Email 發送成功！")
    except Exception as e:
        print(f"❌ Email 發送失敗: {e}")

# =====================================================
# ⚡ 1. 核心邏輯 (IOU Tracker & Utils)
# =====================================================
def compute_iou(box1: List[int], box2: List[int]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0: return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return float(inter_area) / float(union)

class IOUTracker:
    def __init__(self, iou_threshold: float = 0.3) -> None:
        self.iou_threshold = iou_threshold
        self.tracks = {}  # track_id -> box
        self.next_id = 1

    def update(self, detections: List[List[int]], iou_thresh: float) -> List[Tuple[List[int], int]]:
        self.iou_threshold = iou_thresh
        assigned_ids = []
        updated_tracks = {}

        for det in detections:
            best_iou = 0.0
            best_id = None
            for track_id, track_box in self.tracks.items():
                iou = compute_iou(det, track_box)
                if iou > best_iou:
                    best_iou = iou
                    best_id = track_id

            if best_iou > self.iou_threshold and best_id is not None:
                updated_tracks[best_id] = det
                assigned_ids.append((det, best_id))
            else:
                track_id = self.next_id
                self.next_id += 1
                updated_tracks[track_id] = det
                assigned_ids.append((det, track_id))

        self.tracks = updated_tracks
        return assigned_ids

# =====================================================
# 🧠 2. 偵測系統 (NCNN 極速優化版)
# =====================================================
import ncnn # 記得在檔案最上方也要 import ncnn

class DetectionSystem:
    def __init__(self, weights_dir: str = "weights", save_dir: str = "./detection_results"): 
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 初始化 NCNN
        self.net = ncnn.Net()
        # 樹莓派建議關閉 Vulkan 使用 CPU (NCNN 對 ARM CPU 優化極佳)
        self.net.opt.use_vulkan_compute = False 
        
        # 載入模型 (請確認檔名是否正確)

        param_path = os.path.join(weights_dir, "best.param")
        bin_path = os.path.join(weights_dir, "best.bin")
        
        if not os.path.exists(param_path) or not os.path.exists(bin_path):
            st.error(f"❌ 找不到模型檔案！請確認 {weights_dir} 資料夾內有 best_fp16.param 和 .bin")
        
        self.net.load_param(param_path)
        self.net.load_model(bin_path)

        self.iou_tracker = IOUTracker()
        
        # 智慧存檔變數
        self.id_first_seen: Dict[int, float] = {}
        self.id_last_save_time: Dict[int, float] = {}
        self.stay_seconds = 5.0
        self.save_cooldown = 60.0 
        
        # FPS
        self.current_fps = 0.0
        self._last_time = None

        # 🔥 優化關鍵：鎖定輸入尺寸
        self.target_size = 320
        # 正規化參數 (0~255 -> 0~1)
        self.norm_vals = [1/255.0, 1/255.0, 1/255.0]

    def _update_fps(self):
        now = time.time()
        if self._last_time:
            dt = now - self._last_time
            if dt > 0: self.current_fps = 1.0 / dt
        self._last_time = now

    def process_frame(self, frame: np.ndarray, params: dict) -> np.ndarray:
        self._update_fps()
        
        # 參數讀取
        conf_thresh = params['conf']
        iou_thresh = params['iou']
        line_thick = params['thickness']
        font_scale = params['font_scale']
        mosaic_ratio = params['mosaic_ratio']
        
        final_view = frame.copy()
        h_img, w_img = frame.shape[:2]
        
        # 1. 預處理 (Resize + Normalize)
        # NCNN 的 resize 會自動處理 BGR/RGB，這裡輸入是 BGR
        mat_in = ncnn.Mat.from_pixels_resize(
            frame,
            ncnn.Mat.PixelType.PIXEL_BGR,
            w_img, h_img,
            self.target_size, self.target_size
        )
        # 正規化
        mat_in.substract_mean_normalize([], self.norm_vals)

        # 2. 推論
        ex = self.net.create_extractor()
        ex.input("images", mat_in) # 輸入層名稱通常是 images
        ret, mat_out = ex.extract("output0") # 輸出層名稱通常是 output0

        detections = [] # 格式: [x1, y1, x2, y2]
        
        # 3. 後處理 (解碼 YOLO 輸出)
        if ret == 0:
            # YOLOv8 輸出維度: [1, 4+num_classes, num_anchors]
            # 320x320 輸入時，num_anchors = 2100
            # 只有人類別時: [1, 5, 2100] (cx, cy, w, h, score)
            
            # 轉換為 numpy 方便處理
            out_data = np.array(mat_out) # shape (5, 2100)
            
            # 轉置為 (2100, 5)
            out_data = out_data.T 
            
            scores = out_data[:, 4]
            # 篩選信心度
            mask = scores > conf_thresh
            valid_boxes = out_data[mask]
            
            if len(valid_boxes) > 0:
                # 解析座標 (cx, cy, w, h) -> (x1, y1, x2, y2)
                # 需還原回原圖尺寸
                scale_x = w_img / self.target_size
                scale_y = h_img / self.target_size
                
                boxes_xyxy = []
                confidences = []
                
                for row in valid_boxes:
                    cx, cy, w, h = row[:4]
                    score = row[4]
                    # 座標轉換
                    x1 = (cx - w/2) * scale_x
                    y1 = (cy - h/2) * scale_y
                    x2 = (cx + w/2) * scale_x
                    y2 = (cy + h/2) * scale_y
                    
                    boxes_xyxy.append([int(x1), int(y1), int(x2), int(y2)])
                    confidences.append(float(score))
                
                # NMS (非極大值抑制) - 去除重疊框
                indices = cv2.dnn.NMSBoxes(
                    boxes_xyxy, confidences, conf_thresh, iou_thresh
                )
                
                if len(indices) > 0:
                    for i in indices.flatten():
                        detections.append(boxes_xyxy[i])

        # 4. 更新追蹤器 (NCNN 模式強制使用 IOU Tracker)
        # 因為 NCNN 沒有內建 ByteTrack，我們直接用我們寫好的 IOUTracker
        present_ids = []
        tracked = self.iou_tracker.update(detections, iou_thresh)
        
        for box, t_id in tracked:
            present_ids.append(t_id)
            self._draw_box(final_view, box, t_id, line_thick, font_scale)

        # 5. 馬賽克與存檔
        final_view = self._apply_mosaic(final_view, mosaic_ratio)
        self._check_smart_save(present_ids, final_view, params)
        
        return final_view 

# # =====================================================
# # 🧠 2. 偵測系統 (使用 ONNX)
# # =====================================================
# import onnxruntime as ort  # 載入 ONNX Runtime

# class DetectionSystem:
#     def __init__(self, weights_dir: str = "weights", save_dir: str = "./detection_results"): 
#         self.save_dir = Path(save_dir)
#         self.save_dir.mkdir(parents=True, exist_ok=True)
        
#         # 1. 載入 ONNX 模型
#         model_path = os.path.join(weights_dir, "best.onnx")
#         if not os.path.exists(model_path):
#             st.error(f"❌ 找不到模型！請確認 {weights_dir} 資料夾內有 best.onnx")
#             return

#         print(f"🚀 載入 ONNX 模型: {model_path}")
#         # 建立推論 Session
#         self.session = ort.InferenceSession(model_path)
#         # 取得輸入層與輸出層名稱
#         self.input_name = self.session.get_inputs()[0].name
#         self.output_name = self.session.get_outputs()[0].name

#         self.iou_tracker = IOUTracker()
        
#         # 智慧存檔變數
#         self.id_first_seen: Dict[int, float] = {}
#         self.id_last_save_time: Dict[int, float] = {}
#         self.stay_seconds = 5.0
#         self.save_cooldown = 60.0 
        
#         # FPS
#         self.current_fps = 0.0
#         self._last_time = None

#         # 🔥 鎖定輸入尺寸 (必須與訓練時的 imgsz 一致)
#         self.target_size = 320

#     def _update_fps(self):
#         now = time.time()
#         if self._last_time:
#             dt = now - self._last_time
#             if dt > 0: self.current_fps = 1.0 / dt
#         self._last_time = now

#     def process_frame(self, frame: np.ndarray, params: dict) -> np.ndarray:
#         self._update_fps()
        
#         # 參數讀取
#         conf_thresh = params['conf']
#         iou_thresh = params['iou']
        
#         final_view = frame.copy()
#         h_img, w_img = frame.shape[:2]
        
#         # ==========================================
#         # 🔥 ONNX 預處理 (Pre-processing)
#         # ==========================================
#         # 1. Resize: 縮放到 320x320
#         img_resized = cv2.resize(frame, (self.target_size, self.target_size))
        
#         # 2. BGR 轉 RGB (ONNX 模型通常是 RGB 訓練的)
#         img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
#         # 3. 正規化 (0~255 -> 0~1)
#         img_norm = img_rgb.astype(np.float32) / 255.0
        
#         # 4. 維度轉換 (H,W,C) -> (C,H,W)
#         img_transposed = img_norm.transpose(2, 0, 1)
        
#         # 5. 增加 Batch 維度 (C,H,W) -> (1,C,H,W)
#         img_input = np.expand_dims(img_transposed, axis=0)

#         # ==========================================
#         # 🚀 執行推論
#         # ==========================================
#         outputs = self.session.run([self.output_name], {self.input_name: img_input})
        
#         # 解析輸出
#         # YOLOv8 輸出形狀通常是 (1, 5, 2100) -> 需要轉置成 (2100, 5)
#         # 5 代表: [cx, cy, w, h, score]
#         output_data = outputs[0][0].transpose()
        
#         detections = [] # 格式: [x1, y1, x2, y2]
        
#         # 篩選信心度
#         scores = output_data[:, 4]
#         mask = scores > conf_thresh
#         valid_boxes = output_data[mask]
        
#         if len(valid_boxes) > 0:
#             # 計算縮放比例 (將 320 還原回原圖大小)
#             scale_x = w_img / self.target_size
#             scale_y = h_img / self.target_size
            
#             boxes_xyxy = []
#             confidences = []
            
#             for row in valid_boxes:
#                 cx, cy, w, h = row[:4]
#                 score = row[4]
                
#                 # 座標轉換 (cx, cy, w, h) -> (x1, y1, x2, y2)
#                 x1 = (cx - w/2) * scale_x
#                 y1 = (cy - h/2) * scale_y
#                 x2 = (cx + w/2) * scale_x
#                 y2 = (cy + h/2) * scale_y
                
#                 boxes_xyxy.append([int(x1), int(y1), int(x2), int(y2)])
#                 confidences.append(float(score))
            
#             # NMS (非極大值抑制) - 去除重疊框
#             indices = cv2.dnn.NMSBoxes(
#                 boxes_xyxy, confidences, conf_thresh, iou_thresh
#             )
            
#             if len(indices) > 0:
#                 for i in indices.flatten():
#                     detections.append(boxes_xyxy[i])

#         # ==========================================
#         # 追蹤與繪圖
#         # ==========================================
#         present_ids = []
#         tracked = self.iou_tracker.update(detections, iou_thresh)
        
#         for box, t_id in tracked:
#             present_ids.append(t_id)
#             self._draw_box(final_view, box, t_id, params['thickness'], params['font_scale'])

#         # 馬賽克與存檔邏輯 (保留原功能)
#         final_view = self._apply_mosaic(final_view, params['mosaic_ratio'])
#         self._check_smart_save(present_ids, final_view, params)
        
#         return final_view

    def _draw_box(self, img, box, track_id, thick, font_s):
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thick)
        label = f"ID: {track_id}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_s, thick)
        cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_s, (0, 0, 0), thick)

    def _check_smart_save(self, present_ids, frame, params):
        now = time.time()
        present_set = set(present_ids)
        
        # 清除離開的人
        for tid in list(self.id_first_seen.keys()):
            if tid not in present_set:
                self.id_first_seen.pop(tid, None)
                self.id_last_save_time.pop(tid, None)
                
        for tid in present_ids:
            if tid not in self.id_first_seen:
                self.id_first_seen[tid] = now
                continue
            
            # 觸發條件：停留時間 > 設定秒數 且 超過冷卻時間
            if (now - self.id_first_seen[tid] > self.stay_seconds) and \
               (now - self.id_last_save_time.get(tid, 0) > self.save_cooldown):
                
                # 1. 存檔
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                fname = f"ID_{tid}_{timestamp}.jpg"
                file_path = self.save_dir / fname
                cv2.imwrite(str(file_path), frame)
                
                self.id_last_save_time[tid] = now
                print(f"📸 Saved (Protected): {fname}")

                # 2. Email 警報 (非同步)
                if params.get("email_enable", False):
                    # 啟動一個執行緒來寄信，避免卡住畫面
                    email_thread = threading.Thread(
                        target=self._send_alert_worker,
                        args=(str(file_path), tid, timestamp, params)
                    )
                    email_thread.start()

    # 🔥 執行緒工作函數：負責實際寄信
    def _send_alert_worker(self, img_path: str, tid: int, timestamp: str, params: dict):
        print(f"🚀 準備發送 Email 通知: ID {tid}")
        try:
            # 這裡使用您提供的預設值，實際應用建議從 params 讀取更安全
            SMTP_HOST = "smtp.gmail.com"
            SMTP_PORT = 465
            SENDER_USER = "user@gmail.com"
            # 注意：寫在程式碼中的密碼有資安風險，實務上應使用環境變數
            SENDER_PASS = "xxxx xxxx xxxx xxxx".replace(" ", "") 
            
            receiver_email = params.get("email_receiver", SENDER_USER)
            
            subject = f"🚨 警報：發現人員徘徊 (ID: {tid})"
            body = f"""
            系統偵測到有人員在門口徘徊超過設定時間。
            
            - 時間: {timestamp}
            - 追蹤 ID: {tid}
            - 圖片: 詳見附件
            
            這是一封由 YOLO 監控系統自動發送的郵件。
            """
            
            send_email(
                smtp_host=SMTP_HOST,
                smtp_port=SMTP_PORT,
                username=SENDER_USER,
                password=SENDER_PASS,
                sender=SENDER_USER,
                to=[receiver_email],
                subject=subject,
                body_text=body,
                html_body=f"<h3>🚨 監控警報</h3><p>{body.replace(chr(10), '<br>')}</p>",
                attachments=[img_path],
                use_ssl=True
            )
        except Exception as e:
            print(f"❌ 背景寄信執行緒發生錯誤: {e}")

    def _apply_mosaic(self, img, ratio):
        if ratio <= 0: return img
        h, w = img.shape[:2]
        mw = int(w * ratio)
        if mw <= 0: return img
        
        def blur_area(x_start, x_end):
            roi = img[:, x_start:x_end]
            small = cv2.resize(roi, (max(1, (x_end-x_start)//20), max(1, h//20)))
            img[:, x_start:x_end] = cv2.resize(small, (x_end-x_start, h), interpolation=cv2.INTER_NEAREST)
            
        blur_area(0, mw)
        blur_area(w-mw, w)
        return img

# =====================================================
# 🖥️ 3. Streamlit UI
# =====================================================
def list_screenshots(save_dir: Path) -> List[Path]:
    if not save_dir.exists(): return []
    return sorted(save_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)

def main():
    st.set_page_config(page_title="YOLO 居家監控 (Email版)", layout="wide")

    if "detector" not in st.session_state:
        st.session_state.detector = DetectionSystem() # 預設使用 yolo11n.pt，若需 person 模型請在類別內修改
    if "history_idx" not in st.session_state:
        st.session_state.history_idx = 0

    st.title("📹 YOLO 居家偵測系統 + Email 警報")

    with st.sidebar:
        st.header("功能模式")
        app_mode = st.radio("選擇模式", ["🔴 即時偵測", "📂 歷史回放"])
        st.divider()

    # ================= 模式 1: 即時偵測 =================
    if app_mode == "🔴 即時偵測":
        with st.sidebar:
            st.header("⚙️ 偵測設定")
            tracker_mode = st.radio("演算法", ["builtin", "iou"], format_func=lambda x: "ByteTrack (強)" if x == "builtin" else "IOU (快)")
            
            st.subheader("🎯 靈敏度")
            conf_val = st.slider("信心度 (Confidence)", 0.1, 1.0, 0.30, 0.05)
            iou_val = st.slider("重疊門檻 (IOU)", 0.1, 1.0, 0.50, 0.05)
            
            st.subheader("🎨 畫面")
            line_thickness = st.slider("框線粗細", 1, 5, 2)
            font_scale = st.slider("字體大小", 0.5, 2.0, 0.6, 0.1)
            mosaic_ratio = st.slider("馬賽克比例", 0.0, 0.4, 0.15, 0.05)

            st.divider()
            st.header("📧 通知設定")
            enable_email = st.toggle("開啟 Email 警報", value=False)
            email_receiver = st.text_input("收件人信箱", value="user@gmail.com")
            
            run_camera = st.toggle("啟動攝影機", value=True)

        col_video, col_stats = st.columns([3, 1])
        with col_video:
            image_placeholder = st.empty()
        with col_stats:
            status_placeholder = st.empty()

        if run_camera:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if not cap.isOpened():
                st.error("無法開啟相機")
            else:
                while cap.isOpened() and run_camera:
                    if app_mode != "🔴 即時偵測":
                        break
                    
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # 將 Email 設定打包進 params
                    params = {
                        'tracker_type': tracker_mode,
                        'conf': conf_val,
                        'iou': iou_val,
                        'thickness': line_thickness,
                        'font_scale': font_scale,
                        'mosaic_ratio': mosaic_ratio,
                        'email_enable': enable_email,
                        'email_receiver': email_receiver
                    }
                    
                    detector = st.session_state.detector
                    final_img = detector.process_frame(frame, params)
                    image_placeholder.image(final_img, channels="BGR", use_container_width=True)
                    
                    status = f"""
                    ### 📊 狀態監控
                    - **FPS**: `{detector.current_fps:.1f}`
                    - **模式**: `{tracker_mode}`
                    """
                    if enable_email:
                        status += f"\n- **📧 警報**: `開啟`"
                    else:
                        status += f"\n- **📧 警報**: `關閉`"
                        
                    status_placeholder.markdown(status)
                    time.sleep(0.01)
                
                cap.release()
        else:
            image_placeholder.info("攝影機已關閉，請開啟側邊欄開關。")

    # ================= 模式 2: 歷史回放 =================
    elif app_mode == "📂 歷史回放":
        st.subheader("📂 歷史偵測截圖")
        
        save_dir = st.session_state.detector.save_dir
        screenshots = list_screenshots(save_dir)
        
        if not screenshots:
            st.warning("⚠️ 目前資料夾內沒有任何截圖。")
        else:
            total_imgs = len(screenshots)
            if st.session_state.history_idx >= total_imgs:
                st.session_state.history_idx = total_imgs - 1
            
            current_idx = st.session_state.history_idx
            current_file = screenshots[current_idx]
            
            col_img, col_info = st.columns([3, 1])
            with col_img:
                img = cv2.imread(str(current_file))
                if img is not None:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    st.error("圖片讀取失敗")
            
            with col_info:
                st.info(f"檔案: {current_file.name}")
                st.text(f"時間: {datetime.fromtimestamp(current_file.stat().st_mtime)}")
                st.write(f"張數: {current_idx + 1} / {total_imgs}")

                c1, c2 = st.columns(2)
                if c1.button("⬅️ 上一張", use_container_width=True):
                    if current_idx > 0:
                        st.session_state.history_idx -= 1
                        st.rerun()
                if c2.button("下一張 ➡️", use_container_width=True):
                    if current_idx < total_imgs - 1:
                        st.session_state.history_idx += 1
                        st.rerun()

if __name__ == "__main__":
    main()