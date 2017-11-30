# projectfour

This is project four (personal project) I used the german traffic dataset to classify the different traffic signs available. The dataset 
can be downloaded from here : 

http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset

Initially, I have other ideas but due to time constraint had decided on using ready available dataset that has ample examples to help in the 
learning process. It was not clear too whether the emphasis is on learning or on having an accurate model from the instructor.
So I decided on chose this dataset which can give me both assets. 


A Standard CNN model is used and a augmentor library was added to help generate more images to generalised the model. 

From here : https://github.com/mdbloice/Augmentor This augmentor library seems more convinent in customsing the amount of data I need and 
also the type of augmented images can be easily configurable. Which is why I decided to use it. 

After testing a different configuration of cnn architecture I settled on the ones seen in the codes. 
With training accuracy reaching 94 % and testing accuracy reaching 98 %. Both the losses seems low which means it didn't overgeneralised. 
