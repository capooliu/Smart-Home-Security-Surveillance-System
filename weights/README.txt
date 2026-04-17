 檔案說明:

 best.param 和 best.bin -> 優化後轉成ncnn
 best_fp16.param 和 best_fp16.bin -> 優化後轉成ncnn再轉FP16

 在程式170、171行改檔名~

 如果要跑onnx可以改用296-430行的程式(註解起來的那段)