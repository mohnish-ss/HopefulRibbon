# Hopeful Ribbon
## Inspiration
Our inspiration to build such an application was the shortage of doctors across Canada; we realized that separating severe cancer cases from non-severe ones would significantly reduce the workload on doctors and, thus, patiently wait times. Moreover, our team believes in empowering individuals to obtain agency over their health, so we ventured explicitly into breast cancer, as it affects millions of women worldwide.

## What it does
The program aims to have users input data they received from an FNA test, where a needle is inserted, an abnormal lump within the breast and sample cells are extracted. After running simple tests on the cells, raw data is provided, which the user can input into our website. Then, using a machine learning algorithm trained explicitly by a data set, our website will determine whether you are at risk for breast cancer. Finally, the website will send this information to the user via their email, and if they are at risk of cancer, the email will include nearby hospitals.

## How we built it
The back end of the program consists of the machine learning portion, the location service, and the email service, which all use Python. The machine learning portion was coded in PyCharm, using the “tensorflow” library, while the others were coded in Visual Studio Code using Google’s location and email APIs. Finally, the front end (website) was created using HTML and CSS, displaying relevant information regarding our project, i.e. about us, service, etc.

## Challenges we ran into
Some challenges we ran into during the hackathon:
Connecting the back end to the front end using Flask was a more complex challenge than expected. We resolved this by creating a new environment where the application file was in a more accessible directory. 
Merging code was also a problem because this was our first time connecting different parts of code to a single project. We approached this problem by creating communicating with each other about certain variables/methods we would use throughout the code.

## Accomplishments that we're proud of
Our team's main accomplishment was finding the solution to our problem by merging the back and front ends. Initially, we thought the website wouldn’t work after countless hours of trying, but after switching our approach to the problem, we quickly found a solution; after trying for hours on end, finding the answer was a significant breakthrough. In addition, we took great pride in our collaboration throughout the hackathon, as many of our features wouldn’t be possible without teamwork.

## What we learned
We learned many valuable lessons while competing in the hackathon. For example, we learned skills such as connecting the front-end and back-end of an application to create a website. In addition, we also became much more familiar with HTML, CSS and Python and learned many of the intricacies involved in these languages.

## What's next for Hopeful Ribbon
Our team strives to create a better learning model to increase the accuracy at which it interprets and diagnoses patient data. We also strive to expand our service to breast cancer patients and other health conditions such as arrhythmia. 

