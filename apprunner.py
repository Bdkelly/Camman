import subprocess

def main(command):
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"Command executed successfully:\n{result.stdout}")
        if result.stderr:
            print(f"Errors (if any):\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"Error: Command '{command}' not found. Ensure it's in your PATH.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    try:
        command = "python3 -m guiapp.camman"
        main(command)
    except:
        command = "python3 -m guiapp.camman"
        main(command)