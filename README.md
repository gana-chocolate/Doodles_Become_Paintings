# Doodles_Become_Paintings
딥러닝 General Adversarial Network를 활용한 pix2pix 웹 서비스입니다. 
사용자가 간단한 스케치를 넣으면 디테일을 살린 스케치로 변환 후, 그 스케치를 base로 한 이미지가 새로 생성됩니다.

<img width="600" alt="Screen Shot 2020-08-21 at 8 57 26 PM" src="https://user-images.githubusercontent.com/38303729/95704045-992fd100-0c8a-11eb-8ae4-ceba6e227998.png">

ImageCatcher의 욜로 weights 파일 링크 : https://drive.google.com/file/d/1VtA_F3o0rzksVS_Dj64nBlc2luKRCVX-/view?usp=sharing

위치 : ImageCatcher/config

Gan module의 checkpoints 파일 링크 : https://drive.google.com/file/d/1G5b6TNetjZV1gwtdAjc5CHbtp-2_nDRL/view?usp=sharing

위치 : FinalModule/checkpoint

Xception model 파일 링크 : https://drive.google.com/file/d/10MVhnkd9VmDHtuYFefe-j1AnFAgDUgja/view?usp=sharing
https://drive.google.com/file/d/12Jql36h2XvKUZTeGuOfBIww3dGXDFj2b/view?usp=sharing
https://drive.google.com/file/d/1X6Dur5C79362zYTJmIUzur2CF7QnJODz/view?usp=sharing

위치 : FinalModule/xception_module

Doodles_Become_Paintings Server installation

====================================

We also support Windows OS, but we recommend Ubuntu OS.

Set up python
----------------------------------
********************************

    To set up python: http://www.python.org
    $ sudo apt-get install python3


How to install
--------------------------------
**************************

    $ git clone https://github.com/gana-chocolate/Doodles_Become_Paintings.git
    $ cd Doodles_Become_Paintings
    
    ## Please create a virtual environment and install Python package. ##
    
    $ virtualenv -p python3 myvenv
    $ . myvenv/bin/activate
    $ pip install -r requirements.txt

How to run in local
--------------------------------
***************************

    $ cd Doodles_Become_Paintings/Doodles_Become_Paintings
    $ python manage.py runserver
    
        
Contribute
----------------
* Issue Tracker: https://github.com/gana-chocolate/Doodles_Become_Paintings/issues
* Source Code: https://github.com/gana-chocolate/Doodles_Become_Paintings

Contribution guidelines
-----------------------
If you want to contribute to HML, be sure to review the [contribution guideline](https://github.com/gana-chocolate/Doodles_Become_Paintings). This project adheres to HML's code of conduct. By participating, you are expected to uphold this code.

We use GitHub issues for tracking requests and bugs.

License
------------------------
MIT license
