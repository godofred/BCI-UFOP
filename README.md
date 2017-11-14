# BCI-UFOP
Brain-Computer Interface by means of EEG Motor Imagery (beta) - Vinicius Queiroz

This is my Final Undergraduate Project, required for obtaining the title of B.Sc. of Control and Automation Engineering at the Federal University of Ouro Preto (UFOP).
It is not finished, and no guarantee of functionality is provided. The author declares not responsible of misuse of this software.

The MATLAB (*.m) files were the prototype. The Feature Extraction process (which uses Complex Networks and TVEG) still uses MATLAB. The software saves a MATLAB Database (.mat) file with the features extracted, in the format needed for the Machine Learning step.
The Python (*.py) files are the main branch, using TensorFlow (Machine Learning with GPGPU Programming framework) and Qt (GUI) for the Machine Learning step.
We used the EEGMMIDB (https://physionet.org/pn4/eegmmidb/) for all tests.

Installation instructions will be provided in a later date, alongside with instructions on how to use the software and how to alter it to be used with any Database.
Near-future improvements speculations includes adding more Machine Learning Methods (SVM, CNN) and implement the Feature Extraction process in Python.

See the LICENSE file for more information on how this code can be useful without any legal infringements.
