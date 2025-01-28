# Introduction to Machine Learning
**Hebrew University, Jerusalem, Israel**

An introductory code to the world of machine- and statistical learning, aimed for undergraduate students of computer science. The following repository contains:
1) Course Book - Based on lecture- and recitation notes
2) Code examples and graphs generating code, used throughout the book
3) Hands-on guided labs to experience aspects discussed in lectures and recitations
4) Skeleton files for software package ``IMLearn`` developed throughout the course
5) Skeleton files for course exercises


## Setting Up Code and Labs Environment
Set a local copy of this GitHub repository. Do so by [forking and cloning the repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo) or [cloning the repository](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository) using GitBash and 
```
cd LOCAL_REPOSITORY_PATH
git clone https://github.com/GreenGilad/IML.HUJI.git
```

or by downloading and unzipping it in `LOCAL_REPOSITORY_PATH`. Then:

### Anaconda + Jupyter Notebooks Setup
- Download and install Anaconda from [official website](https://www.anaconda.com/products/individual#Downloads). 
- Verify instllation by starting the Anaconda Prompt. A terminal should start with the text `(base)` written at the beginning of the line.
- Set the IML conda environment. Start the Anaconda Prompt and write:
  ```
  conda env create -f "LOCAL_REPOSITORY_PATH\environment.yml"
  ```
  This will create a conda envoronment named `iml.env` with the specifications defined in `environment.yml`. If creating failed due to `ResolvePackageNotFound: plotly-orca` remove this line from environment file, create environment without, and then after activating environment run:
  ```
  conda install -c plotly plotly-orca
  ```
- Activate the environment by `conda activate iml.env`.
- To open one of the Jupyter notebooks:
  ```
  jupyter notebook "LOCAL_REPOSITORY_PATH\lab\Lab 00 - A - Data Analysis In Python - First Steps.ipynb"
  ```

### Using PyCharm
Another option is to run the Jupyter notebooks through the PyCharm IDE plug-in. 
- Install the PyCharm IDE (professional edition) as described on the [Install PyCharm](https://www.jetbrains.com/help/pycharm/installation-guide.html) page. Be sure to install the Jupyter plug-in.
- Follow the [Configure a Conda virtual environment](https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html#conda-requirements) page.
- Open a PyCharm project from `LOCAL_REPOSITORY` and set the project's interpreter to be the `iml.env` environment.

### Using Google Colab
One can also view and run the labs and code examples via Google Colab. It supports loading and running Jupyter notebooks and running using a specified Conda environemnt.





Sure! Here's a more captivating version written in a format that you can directly copy to your GitHub README file:

---

# 🌟 Introduction to Machine Learning  
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

---

### 💡 Ready to Explore?  

Get started with your journey into machine learning today. Let’s build the future—one line of code at a time!  

---

You can now copy this directly into your GitHub README file! Let me know if you'd like to tweak any part of it.
