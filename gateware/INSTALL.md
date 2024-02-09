python -m venv venv
pip install --upgrade pip
#pip install git+https://github.com/amaranth-lang/amaranth.git
# before compat was removed
pip install git+https://github.com/amaranth-lang/amaranth.git@597b1b883924d8949061b52270eb55b97a7cfb76
pip install git+https://github.com/amaranth-farm/amlib
pip install git+https://github.com/amaranth-community-unofficial/adat-core.git
pip install git+https://github.com/amaranth-community-unofficial/amaranth-boards.git
pip install git+https://github.com/amaranth-lang/amaranth-soc.git@87ee8a52d07a2f85b05a04db84644dce48fdfa23
# pip install networkx
cd ../../usb2-highspeed-core
pip install -r requirements.txt
python setup.py install
cd ../adat-usb2-audio-interface/gateware

