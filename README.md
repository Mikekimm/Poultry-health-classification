Author: Michael Kimani
Study: Course in Machine Learning.
Date: February 2026

Project Overview

The project creates an automated poultry health status classifier based on faecal image, which is a crucial issue in Nigerian agriculture, in which small-scale farmers do not always have access to veterinary care. Food security and income in most regions of Nigeria depends greatly on poultry farming, but there is a constant risk of disease outbreaks. The infections can rapidly increase losses when they are transmitted in a flock, with losses sometimes occurring after a few days and eliminating an entire investment. At an early stage, this is therefore very vital yet there is no guarantee of good diagnostic support in a rural society.

The system suggested in this project will work with machine learning which will analyse the pictures of the droppings of poultry and define whether a bird is healthy or probably sick. Alterations in fecal requirements are commonly among the initial apparent signs of contamination. The model is able to detect patterns that could otherwise be missed by someone who is not trained in the field by detecting a subtle visual pattern that might also signify a possible health issue before it can get out of hand and become a massive outbreak. It will further allow farmers to act in time, e.g. isolate the infected birds or get veterinary help to minimise the mortality rates as well as financial losses.

On top of its immediate agricultural application, this work is driven by a larger vision of enhancing the availability of healthcare in Africa, both in Kenya and in other countries. AI can democratise the diagnostic services by incorporating decision support at the expert level into ordinary devices, such as smartphones. Similarly to how AIs can be used in human medical diagnostics, the same can be used to support the animal health systems in resource-limited conditions. This project shows how cutting-edge techniques in machine learning can be applied practically and made available to the regularly ignored communities that high-tech innovation in the future has frequently not addressed.

Problem Statement

Newcastle disease, coccidiosis and other bacterial infections are a major economic loss to Nigeria every year in the poultry industry. Such diseases are easily transmitted among highly populated flocks especially in situations where biosecurity is minimal. In most instances, victims use personal experience and visual observation to diagnose symptoms. Although such an approach can be effective, experienced farmers are not always able to identify warning signs, and this approach is not always effective when scaled.

A number of issues render conventional disease surveillance inadequate. To begin with, the majority of small-scale farmers lack formal veterinary education, which restricts their capacities to differentiate between normal conditions and the onset of a disease. Second, the rural areas often have no easy access to professional diagnostics. Veterinary clinics can be distant to the farms and the cost of consultation can be prohibitive. Third, infections can quickly diffuse in a flock by the time the symptoms are discernible. Lastly, full-time professional supervision is also not economical to most small or medium-sized poultry plants.

Under these limitations, this project examines the possibility of an effective and dependable substitute of machine learning. In particular, it examines the practicality of using image analysis through smartphones as a screening device to poultry health. Should it succeed, a system of this kind can empower farmers and enable them to get instant feedback, lower reliance on limited veterinary resources, and enhance the resilience of the farm in general.

Dataset

The dataset that is utilised in this research is Poultry Birds Poo Imagery Dataset of Health status prediction that was obtained in South-West Nigeria. It occupies 12,000 RGB pictures that are downsampled to 224 by 224 pixels. The data is balanced, 6,000 images are termed healthy and 6,000 termed unhealthy. The labels were determined using a mixture of veterinary examination and farmer experience, which gave a practical ground truth, which was based on practical experience in the field.

Smartphone cameras were used to take photographs in different lighting conditions as well as different backgrounds such as straw bedding, wood shavings, concrete floors, and natural soil. This noise provides a sense of realism in the data and this classification task is closer to the real-life experience of farms. The models do not train on perfectly selected images in the laboratory but rather those data represent agricultural real-life environments.

In order to facilitate strict assessment, stratified sampling was applied in splitting the data into the training, validation, and test sets to maintain balance in the classes. There are 7,680 images (64% of the total) in the training set, 1,920 images (16% of the total) in the accuracy set, and 2,400 images (20% of the total) in the test set. The random seed was fixed at 42 when splitting in order to ensure reproducibility. Having a separate test set is useful to evaluate model performance without any bias as soon as all the training and tuning decisions have been made.

Methodology
Experimental Design

The comparing analysis of three major modelling approaches is carried out in ten experiments to determine which strategy is most effective in this classification task.

The former method is using traditional machine learning algorithms. These are the logistic regression with varying degrees of regularisation, support vector machine with radial basis function kernel, random forest models with different number of trees, and gradient boosting classifiers. Since the traditional models do not directly operate on raw images, every image was transformed into a set of 103 handcrafted features. These characteristics comprise 96 colour histogram values which are the distributions of pixel intensities, five texture statistics, which describe surface patterns, and two shape descriptors, which describe the structural properties.

The second strategy is concerned with deep learning models. The implementation of two convolutional neural network structures was done with the help of the TensorFlow Sequential API. Both models have four convolutional blocks that learn hierarchical visual features directly based on pixel information. Various optimizers were tested to determine their effects on convergence and performance such as Adam and stochastic gradient descent.

The third one is the application of transfer learning based on MobileNetV2 trained on ImageNet. In one setup, the trained base model was frozen and applied only in extracting features. The top twenty layers were then optimised in another set-up to make the network more of an image of poultry. The result of this comparison can be evaluated in the degree of domain-specific adaptation effectiveness.

Technical Implementation
Data Pipeline
In order to provide the efficiency and reproducibility of training, the project provides the work of loading and preprocessing of data through the use of TensorFlow API (tf.data). This pipeline facilitates optimal batching, shuffling, and prefetching and it enhances the use of the gastronomic and minimizes the training duration. Each of the images is scaled to a range of pixel scores [0,1], which provides stability in terms of numbers when updating gradients.
Data augmentation methods are used to enhance generalisation by using the training set. These are random rotation, horizontal flips, slight width and height movements, and zoom transformations. Since images of real-life farms can be different in terms of angle, light and location, augmentation assists the model in acquiring more robust representations as opposed to recalling certain patterns within the training images.
Training Configuration
The binary cross-entropy was used as the loss function to train all deep learning models, which is suitable in two-class classification problems. The batch size was set at 32 which was a compromise between gradient stability and memory efficiency. An early stopping (patience of five epochs) was used to avoid overfitting. When the improvement in the validation performance ceased, the training would automatically stop and reset the most successful weights.
The seed of the whole pipeline was fixed to 42. These contain NumPy operations, TensorFlow weight inception, and scikit-learn train-test splitting. A deterministic behaviour increases the scientific rigour of the experiments and enables the precise replication of the results.
Evaluation Metrics
There were several complementary metrics used to determine model performance. Accuracy is a simple test of general accuracy, which is not entirely indicative of class-specific accuracy. Thus, the precision, recall, and F1-score were also calculated to have a better idea about the trade-off between false positives and false negatives.
Also, threshold-independent performance was assessed with the help of AUC-ROC. This measure is used to determine the performance of the model at all levels of classification between healthy and unhealthy classes. Confusion matrices were utilised to define certain error subsets and possible systematic effects.
Results Summary

Key Findings
Conventional ML Performance: 91% accuracy with handcrafted features, which is decent but the 9 percent error (216 errors made out of 2,400 images) is too high to make it practical to implement disease monitoring.
Deep Learning Benefit: Custom CNNs achieved a 96.25 accuracy, which was 53 percent lower than the error rate of the conventional methods. The slow visual features can be learned automatically and cannot be realised manually.
Transfer Learning Superiority: mobileNetV2 fine-tuning resulted in an accuracy of 98.44 percent (with only 38 errors on 2400 images), indicating that ImageNet pre-training can transfer exceptionally well to images of biology. The frozen base model by itself gave a 97.50 which justified the strength of transfer learning.
Error Analysis: There was no systematic bias in the classes with 38 misclassifications of the best model with equal false positives and false negatives. Typical failure designs included extreme lighting conditions, background interference and borderline cases in which even the human experts could disagree.
Practical Impact: An error rate of 9% was reduced to 1.56% and 219 fewer mistakes daily would have been prevented in a 10,000-bird farm, which could prevent disease outbreak and save a lot of money.

git clone https://github.com/Mikekimm/Poultry-health-classification.git
Project Structure

poultry-health-classification/
├── notebook9231e18691.ipynb # Main Jupyter notebook with all experiments
├── experiment_results.csv # Comprehensive results table
├── README.md # This file
└── report/ # (Optional) Written report and documentation

Requirements
Python

Core Libraries
TensorFlow for deep learning framework
Scikit-learn for Traditional ML algorithms
Numpy, Pandas for data manipulation
Matplotib, Seaborn for visualization
OpenCV, Pillow for image processing

For Installing:
pip install tensorflow scikit-learn numpy pandas matplotlib seaborn opencv-python pillow

How to Run
Kaggle ( The best one to use)

1. Upload the notebook to kaggle
2. Upload the poultry birds poo imagery dataset
3. Enable GPU accelerator
4. Update the dataset path in cell 5 to match your kaggle data set location
5. Run all cells

Collab
1. Upload the notebook to Google Collable
2. Upload the dataset to Google Drive
3. Update the DATASET_PATH in cell 5 to match your Drive Structure
4. Enable GPU runtime
 5. Run all cells

The notebook automatically detects the platform whether it is Kaggle, Collab or Local and adjusts its path accordingly.
 
Notebook Structure
The notebook consists of 8 major categories:
1.Setting up data and loading it:
2. Exploratory Data Analysis
3. Data Preprocessing and Feature Engineering
4. Classical Machine Learning Models
5. Deep Learning Models
6. Summary Experiment Tracking Table
7. Final Comparison of Model and Error Analysis
8. Key Findings and Insights





Performance Comparison
The experiment outcomes show an evident distinction in modelling methods.
The results of traditional machine learning were decent, with the optimal model being a Random Forest with 200 trees. This setup had a test accuracy of 91.04 and AUC of 96.85. These findings indicate that handcrafted colour, texture, and shape characteristics have significant diagnostic content. Nonetheless, the other error rate of around 9 percent is equal to 216 false classifications among 2,400 test images which might be excessive to be used in disease surveillance on a large scale level.
Conventional convolutional neural networks enhanced the performance significantly. The CNN using Adam optimizer had an accuracy of 96.25 percent in the test and the AUC of 99.27. This is a 53% error reduction compared to traditional machine learning. The CNN automatically learnt the hierarchical visual representations instead of using manually designed features. Such depictions have the ability to encode delicate mixtures of gradients, textures and space structures that can not be described by handcrafted features.
Transfer learning achieved the best results. The optimised MobileNetV2 model obtained the test accuracy of 98.44 and the AUC of 99.67. This is just 38 misclassifications in 2400 test samples. The frozen model at the base-level was able to retrieve 97.50% accuracy without fine-tuning, which points to the impressive generalisation capacity of ImageNet pre-training. These results indicate that visual attributes acquired on large-scale natural image tests are applicable to specialised biological visuals.
Error Analysis
Careful examination of the 38 misclassifications made by the best-performing model did not show an intense bias of classes. The distribution of false positives and false negatives was balanced, indicating that there was no systemic bias of the model toward one of the classes.
A lot of the mistakes were in the adverse lighting factors like strong shadows or overexposure. There are occasions when intricate backgrounds have caused obstruction to clear feature extraction. Moreover, borderline cases, in which the differences in the visuals were slight, were challenging even to human evaluators. This could imply that the performance of the model can already begin to reach the visual-only classification limits on some ambiguous samples.
To improve in future, it is necessary to understand these failure modes. It may be possible to reduce the remaining errors by improving preprocessing methods, adding attention-based methods, or using contextual farm information.
Practical Impact
The error rate of 9 percent was dropped to 1.56 percent, which has great real-life implications. Even a small percentage increase in the accuracy of the diagnostics in a farm with 10,000 birds can result in hundreds of correctly determined cases per day. The early detection enables the farmers to quarantine infected birds promptly, treat them where necessary, and avoid massive outbreaks.
In addition to economical savings, better disease surveillance ensures animal welfare and helps to have more stable food chains. With the integration of this system into an application installed into a smartphone, farmers will be able to reach the level of screening referred to as experts without the necessity to be under constant veterinary control. This democraticization of diagnostic power is also in tandem with other more general objectives of technological equity.
Real-World Applications
The system has viable advantages to various stakeholders.
In the case of farmers, it is giving them early diagnosis of a disease before it manifests in its serious form. This will lessen the loss of money and emergency interventions. Since the system uses a smartphone camera only, it can be used even in low-resource or rural contexts.
In the case of agricultural extension services, the model can be used to provide scalable screening between multiple farms. Predictive data in the aggregate could be useful in monitoring disease prevalence patterns and instructions to specific interventions. The high risk regions can then be served more efficiently by veterinary resources.
In the perspective of researchers, the project forms a baseline of going beyond the binary classification to multi-class disease detection. The methodology also shows how transfer learning is also applicable in special agricultural fields with restricted datasets.
Limitations and Future Work
Although performing successfully, the project has a number of constraints. The dataset is also geographically localised on South-West Nigeria, which might not be generalised to other climates, poultry breeds or feeding procedures. Large-scale deployment faces the need to be externally validated using data in other areas.
The dichotomous classification system makes the issue simple as healthy/unhealthy. Although it can be effective in screening, it lacks the ability to discriminate certain diseases. Future models must be extended to the multi-class classification to offer more actionable information.
There are also issues with image constraints. Each and every sample was standardised to 224 using 224 pixels, which might not reflect the variability of smartphone images in the real world completely. The model should be tested on the work in the future using raw and unprocessed images that are taken in natural conditions.

The other limitation is explainability. The model is highly accurate, however, it does not provide a clear indication of what visual features contributed to each prediction. Adding explainable AI, like Grad-CAM, would raise the levels of trust and adoption among the users. Visualisation map maps would enable the farmers and agricultural officers to view what areas of an image were most important to the model in the classification of a sample as either healthy or unhealthy. This would bring about increased openness to the system and would more closely resemble a group decision-support system rather than a black box. Trust is equally significant as the raw accuracy in the real-world deployment. When users know the reason a prediction has been made, they will tend to trust it and use it as a part of their lives.
Long-term directions are longitudinal tracking of individual birds, combining the data of environmental sensors, cross-regional validation, quantization, and pruning methods of deployment on lower-end smartphones. The longitudinal tracking would help the system to track the health condition of a bird in a time span and not just a single snapshot of an image. Minor differences that would seemingly be not important on their own can be interpreted as significant when one observes a progress. With the introduction of the temporal awareness into the system, early warnings may also be even more reliable.
Another possible improvement is the addition of environmental data. Temperature, humidity, quality of ventilation, feed composition, and stocking density are factors that determine the health of poultry. This would be a more holistic health assessment model when the visual predictions are combined with the context sensor data. As an illustration, in case the sensors in the environment report that there is uncharacteristically high moisture in the air, and the visual model shows that there are some early warning indicators, the integrated system might produce more powerful alerts. Such a multi-modal strategy can help to decrease the false positives and enhance predictive strength in adverse conditions.
It is also important that cross-regional validation takes place. Even though the dataset employed in this project is realistic in the farm conditions of South-West Nigeria, the conditions of agriculture vary tremendously in different regions. Differences in weather, type of chicken breeds, type of feed used and type of housing can vary visual features. The model should be evaluated on new datasets gathered immediately before large-scale deployment in other states in Nigeria and preferably in other African nations. This validation would guarantee that the system can be extended to areas outside of its training environment and will not be biassed against certain local conditions inadvertently.
Another important priority is the model optimization of mobile deployment. Although MobileNetV2 is already lean and efficient, additional optimization methods can be used to make it more applicable in the real world. Quantization also makes models much smaller by quantizing floating-point weights, which significantly reduces the memory and inference time. Redundant or less important network connexions are eliminated by pruning more computation. Such methods are particularly relevant in rural settings where farmers can be taking advantage of affordable smartphones with low processing power and storage capacity.
Other than technical enhancements, deployment strategy should also be keenly handled. An easy-to-use mobile application would require straightforward interface, automatic indication of prediction, and guidance on the course of next action. As an example, the app might not show a probability score, but it can be classified into easily understandable risk levels, e.g., Low Risk, Medium Risk, or High Risk with recommended actions. Effective communication will minimise confusion and make predictions relevant decisions.
The successful adoption would also be achieved with the help of training and awareness programmes. The farmers might also be reluctant towards AI-driven tools, particularly when they have years of years of doing things in a certain way. Trust-building could take place through demonstration sessions, pilot deployments and working with local agricultural extension officers. Once the farmers see the system successfully detecting early instances of illness, they will have no problem developing confidence in the technology, naturally.
There should also be data privacy and ownership. The farmers ought to be in control of the images taken in their farms. The central storage of any research or monitoring data should be transparent and secure. The creation of explicit data governance policies both would safeguard the interests of users and would allow the adoption of the model to be enhanced further by way of aggregated learning.
Research wise, it is significant to take the classification model a step further and incorporate several categories of diseases. The future models would not necessarily classify samples as healthy or unhealthy because it was able to distinguish between Newcastle disease, coccidiosis, bacterial infections and nutritional disorders. The multi-class classification would be more specific, and farmers can more effectively choose the correct treatment or preventive measures.
The other direction is promising, which is semi-supervised or self-supervised learning. Harvesting and sorting farm data may be a time-consuming and costly procedure. Self-supervised pre-training on large quantities of unlabeled data might lessen the need to use manual annotation and still achieve high performance. This plan would prove especially helpful when expanding the system to new areas that have low labelled data.
Federated learning may also be considered as a privacy-preserving training. Models might be trained on single devices and then aggregated on a global scale instead of sending the images of farms to a central server. This would ensure that sensitive farm information is secured and that the model performance keeps on being enhanced in distributed environments.
Ethical issues are the focus during development and deployment. Although AI systems could be used to improve productivity and disease control, it is advisable that it should be used in addition to human expertise and not to substitute it. The veterinarians and agricultural experts are still needed to verify diagnoses and give prescriptions. The system is also supposed to be a screening tool and not an ultimate medical authority.
Within the greater framework, this project will lead to a vision of empowerment through technology. The way it refines the tools of the latest machine learning to address the real life issues related to agriculture reveals that innovation does not necessarily have to be concentrated in the urban environment with a large number of resources. Problems can and must be planned to accommodate the rural folks. The same concepts used here transfer learning, efficient model design, reproducibility and user-centred deployment can be applied to other applications like crop disease detection, livestock monitoring as well as human-health diagnosis.
Finally, the effectiveness of this system must not be evaluated only through the metrics of test accuracy but on its practical effects. When it helps farmers to identify disease earlier, minimise economic losses and enhance food security, then it serves its purpose. The future of the technology is whether it can be developed into a research prototype to make it an agriculture tool; it can be eventually rolled out nationwide with continuous improvement and community participation.

Acknowledgments
Instructor: Dirac Murairi
Dataset Creators: Scholars who assembled the Poultry Birds Poo Imagery Dataset in South-West Nigeria.
This is an educational project that is developed during a Machine Learning course assignment. The data is taken by its initial licensing conditions.
Contact
Michael Kimani
Machine Learning Studen

Link to my video 
https://youtu.be/6yDhGvgi4Q8
