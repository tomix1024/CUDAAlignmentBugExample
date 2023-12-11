# Reproduce a CUDA Bug regarding Variable Alignment

This is a (minimal) example of triggering a bug in CUDA related to variable alignment.
The `main_kernel()` should (indirectly) create an instance of `NEEData`, modify the first member `state`, and write the result into the `data` buffer.
The failure case looks as follows:
The updated value of the `state` variable is *not* written into the first four bytes of the `data` buffer, but the default value is.

### System
Tested on Ubuntu 22.04 with the latest `nvidia-driver-545` and `cuda-12-3` installed running the following GPUs: RTX 2080, RTX 3090, RTX 4070M.

### Expected Output
```bash
$ ./run.sh
reached exit point
data: 0x1 0xffffffff 0x0 0x0 0x3 0xffffffff 0xffffffff 0xffffffff
```

### Actual Output
```bash
$ ./run.sh
reached exit point
data: 0x2 0xffffffff 0x0 0x0 0x3 0xffffffff 0xffffffff 0xffffffff
```

There are various `#defines` commented out in `debug.cu`, each of them will resolve the issue, but the original code should work as well.
