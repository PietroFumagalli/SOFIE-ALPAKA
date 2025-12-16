import subprocess
import sys
import os

def build_kernel_tests():
    """
    Calls the Makefile to build the kernel tests.
    Returns True if successful, False otherwise.
    """
    print("Building project with Make...")
    try:
        # Check if Makefile exists
        if not os.path.exists("Makefile"):
            print("Error: Makefile not found in current directory")
            return False

        # Run 'make'.
        # capture_output=False lets the user see the compiler output in real-time
        subprocess.run(["make"], check=True)
        
        print("✅ Build successful\n")
        return True
        
    except subprocess.CalledProcessError:
        print("❌ Build failed. Please fix C++ errors before running benchmarks.")
        return False
    except FileNotFoundError:
        print("❌ Error: 'make' command not found. Is it installed?")
        return False

def run_benchmark(executable_path, args):
    """
    Runs the compiled executable with arguments.
    """
    if not os.path.exists(executable_path):
        print(f"❌ Error: Executable '{executable_path}' not found after build.")
        return

    print(f"🚀 Running {executable_path} with args: {args}...")
    try:
        # Construct the command
        cmd = [executable_path] + [str(a) for a in args]
        
        # Run and capture output for parsing
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("--- Output ---")
        print(result.stdout)
        
        # TODO: Add your parsing logic here (regex or string split) to get the time
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Execution failed with return code {e.returncode}")
        print("Stderr:", e.stderr)

if __name__ == "__main__":
    # Build Phase
    if not build_kernel_tests():
        sys.exit(1)

    # Benchmark Phase
    # Adjust this path to match where your Makefile outputs the binary
    binary_path = "./build/alpaka_test_kernel" 
    
    input_sizes = [1024, 2048, 4096]
    
    for size in input_sizes:
        run_benchmark(binary_path, [size])
