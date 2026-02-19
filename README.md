# Principal Component Analysis (PCA) from Scratch 🎯

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-orange.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> *A deep dive into dimensionality reduction using eigenvalues, eigenvectors, and covariance matrices - all built from scratch!*

---

## 👋 Hey There!

Welcome to my PCA implementation project! I'm Michael Kimani, and this is my take on one of the most powerful techniques in machine learning and data science. Instead of just using a library, I built PCA from the ground up to really understand what's happening under the hood.

**Why should you care about this project?**
- 🧠 It's educational - see exactly how PCA works, step by step
- 🌍 It's meaningful - uses real African health data
- 💪 It's practical - includes performance optimization and validation
- 📊 It's visual - lots of graphs to help you understand the concepts

---

## 🎯 What is PCA Anyway?

Imagine you're trying to describe a person. You could use hundreds of measurements (height, weight, eye color, favorite food, shoe size, etc.), but really, just a few key characteristics capture most of what makes them unique. That's PCA!

**Principal Component Analysis** is a technique that:
- Takes high-dimensional data (lots of features)
- Finds the most important patterns
- Reduces dimensions while keeping the essential information
- Makes data easier to visualize and analyze

**Real-world applications:**
- 📸 Face recognition systems
- 🎬 Movie recommendation engines (Netflix, anyone?)
- 🧬 Genetic research
- 📈 Financial risk assessment
- 🗣️ Speech recognition

---

## 📚 What I Learned

This project was a journey through some seriously cool math:

### 1. **Covariance Matrices** 🔢
Understanding how features relate to each other. If GDP goes up, does healthcare improve? The covariance matrix tells us!

$$\text{Cov}(X) = \frac{1}{n-1}X^TX$$

### 2. **Eigenvalues & Eigenvectors** 🎓
These aren't just abstract math concepts - they're the secret sauce of PCA:
- **Eigenvalues**: Tell us how much variance each component explains
- **Eigenvectors**: Point us in the direction of maximum variance

$$\text{Cov}(X)\mathbf{v} = \lambda\mathbf{v}$$

### 3. **Dimensionality Reduction** 📉
Taking 15 features and compressing them down to 3-4 components while keeping 95% of the information. Magic? Nope, just good math!

---

## 🚀 Project Structure

```
alu-webstack/
│
├── PCA_Implementation.ipynb    # The main notebook with everything
├── README.md                    # You're reading it!
└── data/                       # African health & development data
```

---

## 🎨 What's Inside the Notebook?

### Part 1: Data Preparation 🧹
- Loading African health indicators
- Handling missing values (because real data is messy!)
- Encoding categorical features
- Standardizing features (super important for PCA!)

### Part 2: PCA from Scratch 🔨
**Task 1: The Core Implementation**
- Computing the covariance matrix
- Finding eigenvalues and eigenvectors
- Projecting data onto principal components

**Task 2: Smart Component Selection 🧠**
- Dynamic selection based on explained variance
- Answering: "How many components do I actually need?"
- Tested with multiple thresholds (80%, 85%, 90%, 95%, 99%)

**Task 3: Performance Optimization ⚡**
- Vectorized NumPy operations (no loops!)
- Benchmarking on datasets of different sizes
- Making it fast enough for real-world use

### Part 3: Validation & Visualization 📊
- Comparing with scikit-learn's PCA
- Beautiful 2D and 3D visualizations
- Feature importance analysis
- Interactive plots showing regional patterns

---

## 📊 The Dataset

I'm using **African Health and Development Indicators** with:
- 200 observations (countries/regions)
- 15 features including:
  - Life expectancy
  - GDP per capita
  - Infant & maternal mortality rates
  - Access to clean water
  - Literacy rates
  - Healthcare expenditure
  - HIV & malaria prevalence
  - And more!

**Why this dataset?**
1. It's **meaningful** - represents real challenges and progress in Africa
2. It's **complex** - features are correlated in interesting ways
3. It's **real** - includes missing values and mixed data types
4. It's **relevant** - not your typical Iris or Wine dataset!

---

## 💡 Key Results

Here's what I discovered:

- **Dimensionality Reduction**: Compressed 15 features down to just **4 components**
- **Information Retained**: Kept **95%** of the original variance
- **Regional Patterns**: Clear separation between North, West, East, Central, and Southern Africa
- **Performance**: Processes 5,000 samples in under a second
- **Accuracy**: Results match scikit-learn within numerical precision

### Most Important Findings:
1. **PC1** captures overall development (health + economic indicators)
2. **PC2** separates urban vs rural characteristics
3. **PC3** highlights disease burden differences
4. Regional clustering is clearly visible in the PCA space

---

## 🛠️ Technologies Used

- **Python 3.8+**: The language of data science
- **NumPy**: For all the heavy mathematical lifting
- **Pandas**: Data manipulation made easy
- **Matplotlib & Seaborn**: Creating beautiful visualizations
- **Scikit-learn**: For validation (and a reality check!)

---

## 🚀 Getting Started

### Prerequisites
```bash
python >= 3.8
numpy
pandas
matplotlib
seaborn
scikit-learn
```

### Installation
```bash
# Clone the repository
git clone https://github.com/mikekimm/alu-webstack.git
cd alu-webstack

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn

# Launch Jupyter Notebook
jupyter notebook PCA_Implementation.ipynb
```

### Running the Code
Simply open the notebook and run all cells! Each section is well-documented and includes:
- Explanations of what's happening
- Mathematical formulas (in LaTeX)
- Code comments
- Output visualizations

---

## 📈 Sample Visualizations

The notebook includes:
- 📊 **Scree plots** showing variance explained by each component
- 🎨 **2D & 3D scatter plots** of transformed data
- 🔥 **Heatmaps** of covariance matrices and feature loadings
- 📉 **Performance benchmarks** across different dataset sizes
- 🌍 **Regional clustering** visualizations

---

## 🎓 What I Learned (The Real Stuff)

Beyond the technical skills, this project taught me:

1. **Math isn't scary**: Eigenvalues seemed intimidating, but they're just telling us which directions in our data matter most
2. **Standardization matters**: Forgot to standardize once - PCA went wild! Features with bigger scales dominated everything
3. **Visualization is key**: You can have perfect math, but if you can't show it, nobody will understand
4. **Validation is crucial**: Comparing with scikit-learn caught subtle bugs I would've missed
5. **Real data is messy**: Missing values, mixed types, outliers - dealing with these made the project so much better

---

## 🤔 Challenges I Faced

**Challenge 1: Understanding Eigenvectors**
- *Problem*: The math notation was confusing
- *Solution*: Drew lots of diagrams, watched videos, and eventually it clicked!

**Challenge 2: Numerical Stability**
- *Problem*: Sometimes eigenvalues came out slightly negative (should be impossible!)
- *Solution*: Learned about floating-point precision and added proper validation

**Challenge 3: Making It Fast**
- *Problem*: Initial implementation was slow
- *Solution*: Vectorized everything with NumPy - 10x speedup!

---

## 📝 Assignment Requirements Checklist

✅ **Data Requirements**
- [x] More than 10 columns (15 features)
- [x] Missing values present
- [x] At least one non-numeric column
- [x] Impactful African data (not generic datasets)

✅ **Task 1: PCA Implementation**
- [x] Covariance matrix calculation
- [x] Eigenvalue computation
- [x] Eigenvector extraction
- [x] Data projection onto principal components

✅ **Task 2: Dynamic Component Selection**
- [x] Automatic selection based on explained variance
- [x] Threshold-based approach (95% variance)
- [x] Tested with multiple thresholds

✅ **Task 3: Performance Optimization**
- [x] Vectorized implementation (no loops)
- [x] Benchmarking across different sizes
- [x] Memory-efficient computation
- [x] Scalability testing

✅ **Additional Requirements**
- [x] All cell outputs visible
- [x] Code well-documented
- [x] Mathematical formulas included
- [x] Comprehensive visualizations

---

## 🔮 Future Improvements

If I continue this project, I'd love to:
- [ ] Add kernel PCA for non-linear dimensionality reduction
- [ ] Implement incremental PCA for streaming data
- [ ] Create an interactive dashboard with Plotly
- [ ] Compare with other techniques (t-SNE, UMAP)
- [ ] Apply to more African datasets (economics, education, agriculture)

---

## 🤝 Contributing

This is a learning project, but I'm always open to suggestions! If you:
- Spot a bug 🐛
- Have an idea for improvement 💡
- Want to discuss the math 🧮
- Have cool African datasets to try 🌍

Feel free to open an issue or reach out!

---

## 📚 Resources That Helped Me

- [StatQuest PCA Video](https://www.youtube.com/watch?v=FgakZw6K1QQ) - Best explanation ever!
- [3Blue1Brown Linear Algebra Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) - Made me love linear algebra
- [Scikit-learn PCA Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- *Introduction to Statistical Learning* - The theory behind it all

---

## 📄 License

This project is licensed under the MIT License - feel free to use it for learning!

---

## 👨‍💻 About Me

Hi! I'm **Michael Kimani**, a student passionate about:
- 🤖 Machine Learning & AI
- 📊 Data Science
- 🌍 Using tech to solve African challenges
- 🎓 Understanding the math behind the magic

**Connect with me:**
- GitHub: [@mikekimm](https://github.com/mikekimm)
- Repository: [alu-webstack](https://github.com/mikekimm/alu-webstack)

---

## 🙏 Acknowledgments

- My Linear Algebra professor for the inspiration
- The open-source community for amazing tools
- My classmates for the study sessions
- Coffee ☕ for keeping me awake during debugging

---

## 📞 Questions?

If you have questions about:
- How PCA works
- The implementation details
- The African health data
- Anything else in this project

Feel free to open an issue! I love discussing this stuff. 😊

---

<div align="center">

**Made with ❤️ and lots of ☕ by Michael Kimani**

*October 2025*

⭐ If you found this helpful, consider starring the repo!

</div>
