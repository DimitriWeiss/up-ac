from sys import platform
import wget
import zipfile
import os
import shutil
import stat


def get_OAT():
    # Set path to up-ac
    path = os.getcwd().rsplit('up_ac', 1)[0]
    if path[-1] != "/":
        path += "/"
    path += 'up_ac'

    # Path to OAT directory
    save_as = f'{path}/OAT/OAT.zip'

    # Check platform
    if platform in ("linux", "linux2"):
        url = 'https://docs.optano.com/algorithm.tuner/current/' + \
            'OPTANO.Algorithm.Tuner.Application.2.1.0_linux-x64.zip'
    elif platform == "darwin":
        url = 'https://docs.optano.com/algorithm.tuner/current/' + \
            'OPTANO.Algorithm.Tuner.Application.2.1.0_osx-x64.zip'
    elif platform == "win32":
        url = 'https://docs.optano.com/algorithm.tuner/current/' + \
            'OPTANO.Algorithm.Tuner.Application.2.1.0_win-x64.zip'

    # Make OAT directory in up-ac
    if not os.path.isdir(f'{path}/OAT'):
        os.mkdir(f'{path}/OAT')

    # Download binaries, if not in up-ac/OAT
    if not os.path.isfile(f'{path}/OAT/OAT.zip'):
        wget.download(url, out=save_as)

    # Unzip and delete .zip
    if os.path.isfile(f'{path}/OAT/OAT.zip') and \
            not os.path.isfile(
                f'{path}/OAT/Optano.Algorithm.Tuner.Application'):

        with zipfile.ZipFile(f'{path}/OAT/OAT.zip', 'r') as zip_ref:
            zip_ref.extractall(f'{path}/OAT')
        if os.path.isfile(f'{path}/OAT/OAT.zip'):
            os.remove(f'{path}/OAT/OAT.zip')

    st = os.stat(f'{path}/OAT/Optano.Algorithm.Tuner.Application')
    os.chmod(f'{path}/OAT/Optano.Algorithm.Tuner.Application',
             st.st_mode | stat.S_IEXEC)


def copy_call_engine_OAT():
    # Copy call_engine_OAT from utils to OAT directory
    path = os.getcwd().rsplit('up_ac', 1)[0]
    if path[-1] != "/":
        path += "/"
    path += 'up_ac'
    
    shutil.copy(f'{path}/utils/call_engine_OAT.py', f'{path}/OAT/') 


def delete_OAT():
    # Set path to up-ac
    path = os.getcwd().rsplit('up_ac', 1)[0]
    if path[-1] != "/":
        path += "/"
    path += 'up_ac'
    if os.path.isdir(f'{path}/OAT/'):
        shutil.rmtree(f'{path}/OAT/')
