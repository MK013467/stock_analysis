import subprocess
import sys
def install_requirements(requirements_path = "requirements.txt"):
    '''

    :param requirements_path:
    :return:
    '''

    try:
        subprocess.check_call([sys.executable, '-m', "pip", "install" , "-r" , requirements_path])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages from {requirements_path}. Error: {e}")

if __name__ == '__main__':
    install_requirements()
