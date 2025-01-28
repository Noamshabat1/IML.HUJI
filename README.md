<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Hebrew_University_Logo.svg/1200px-Hebrew_University_Logo.svg.png" alt="huji-logo" height="150px" />
  <h1 align="center" style="border-bottom: none"><b> Introduction to Machine Learning</b></h1>

**Hebrew University, Jerusalem, Israel**  

Welcome to the **Introduction to Machine Learning** repository! This course is your gateway to the exciting world of machine learning and statistical learning. Whether you're diving into cutting-edge algorithms or building your own tools, this repository provides everything you need to succeed.  

### What’s Inside?  
1️⃣ **Course Book**: Comprehensive notes from lectures and recitations.  
2️⃣ **Code Examples & Graphs**: Ready-to-run scripts used throughout the course book.  
3️⃣ **Guided Labs**: Hands-on labs to apply what you’ve learned.  
4️⃣ **`IMLearn` Skeleton Files**: Start developing your own software package.  
5️⃣ **Exercise Skeletons**: Templates for course assignments.  

---

## 🚀 Getting Started  

To get started, make a local copy of this repository:  

1. **Option 1: Fork and Clone**  
   [Fork this repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo) and clone it to your local machine:  
   ```bash
   cd LOCAL_REPOSITORY_PATH  
   git clone https://github.com/GreenGilad/IML.HUJI.git  
   ```  

2. **Option 2: Download**  
   [Download the repository](https://github.com/GreenGilad/IML.HUJI/archive/refs/heads/main.zip) as a ZIP file and extract it to `LOCAL_REPOSITORY_PATH`.  

---

### 🛠 Setting Up with Anaconda  

1. **Install Anaconda**:  
   Download and install Anaconda from the [official website](https://www.anaconda.com/products/individual#Downloads).  

2. **Verify Installation**:  
   Open the Anaconda Prompt. You should see `(base)` at the start of the command line.  

3. **Create the Conda Environment**:  
   Run the following command in the Anaconda Prompt:  
   ```bash
   conda env create -f "LOCAL_REPOSITORY_PATH/environment.yml"  
   ```  
   If you see `ResolvePackageNotFound: plotly-orca`, remove the `plotly-orca` line from `environment.yml` and rerun the command. Then, install it manually:  
   ```bash
   conda install -c plotly plotly-orca  
   ```  

4. **Activate the Environment**:  
   ```bash
   conda activate iml.env  
   ```  

5. **Run Jupyter Notebooks**:  
   Open a lab notebook, for example:  
   ```bash
   jupyter notebook "LOCAL_REPOSITORY_PATH/lab/Lab 00 - A - Data Analysis In Python - First Steps.ipynb"  
   ```

---

### 🔧 Using PyCharm  

You can also run the course notebooks through PyCharm, the professional-grade IDE.  

1. **Install PyCharm**:  
   Follow the [installation guide](https://www.jetbrains.com/help/pycharm/installation-guide.html). Make sure to enable the Jupyter plug-in.  

2. **Configure Conda Environment**:  
   Set up the `iml.env` environment in PyCharm as described in the [Conda support documentation](https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html#conda-requirements).  

3. **Open the Repository**:  
   Load the repository folder as a PyCharm project and set the interpreter to `iml.env`.  

---

### 🌐 Using Google Colab  

Want to code in the cloud? You can run labs and examples via [Google Colab](https://colab.research.google.com), which supports Jupyter notebooks and Conda environments for seamless execution.  
