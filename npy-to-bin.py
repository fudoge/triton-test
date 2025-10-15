import numpy as np
x = np.load("example_input.npy").astype(np.float32)
x.tofile("example_input.bin")  # raw binary, header 없음
