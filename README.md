# BCI-UFOP
Brain-Computer Interface by means of EEG Motor Imagery (beta) - Vinicius Queiroz

This is my Final Undergraduate Project, required for obtaining the title of B.Sc. in Control and Automation Engineering at the Federal University of Ouro Preto (UFOP). It consists with a set of code files for pre-processing of EEG signals using the TVEG with Complex Networks method and training of an offline classifier for a Motor Imagery BCI.
This project is in its first steps, is unfinished to the date, and no guarantee of functionality is provided. The author declares not responsible for any misuse of this software.

The MATLAB (.m) files were the prototype. The Feature Extraction process (which uses Complex Networks and TVEG) still uses MATLAB. The software saves a MATLAB Database (.mat) file with the features extracted, in the format needed for the Machine Learning step.
The Python (.py) files are the main branch, using TensorFlow (Machine Learning with GPGPU Programming framework) and Qt (GUI) for the Machine Learning step.
We used the EEGMMIDB (https://physionet.org/pn4/eegmmidb/) for all tests.

Installation instructions might (eventually) be provided in a later date, alongside with instructions on how to use the software and how to alter it to be used with any Database.

See the LICENSE file for more information on how this code can be useful without any legal infringements.
