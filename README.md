# 🧠 Ethical AI and Future Trends Workshop

## 📂 GitHub Wiki Structure  
The wiki for this workshop follows a structured format to maintain clarity and accessibility:  

- 📜 **Overview:** Introduction to the workshop and objectives  
- 💻 **Using a Local Computer:** Setting up and running Docker on a personal machine  
- 🏢 **Using a Supercomputer (HPC):** Steps for running the workshop on an HPC system  
- 🏛️ **Legacy Content:** Archived or previous versions of workshop material  

---

## 📜 Overview  

## 💻 Using a Local Computer  

### 🔧 Step 1: Open Terminal and Configure System  

#### Windows (WSL)  
- Install WSL:  
  ```sh
  wsl --install
- Set up a Linux Distribution:
  ```sh
  wsl --set-default-version 2
- Open WSL Terminal:
  ```sh
  Press Win + R, type wsl, and hit Enter.

#### MacOS
- Open Terminal:
  ```sh
  Press Command + Space, type Terminal, and hit Enter.
- Ensure Homebrew is Installed:
  ```sh

  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

#### 🐧 Linux
- Open Terminal:
  ```sh
  Press Ctrl + Alt + T.
- Ensure Package Manager is Updated:
  ```sh
  sudo apt update && sudo apt upgrade -y

### 🔧 Step 2: Build and Use the Docker Image
- A pre-configured Docker image has been created to simplify the setup process.
- 🛠 Docker Image Name: intro_to_hpc_ai_jupyter

#### 📥 Steps to Use the Docker Image
- Pull the Docker Image:
  ````sh
  docker pull ud31/intro_to_hpc_ai_jupyter
- Run the Docker Container:
  ```sh
  docker run -p 8888:8888 your-repo/intro_to_hpc_ai_jupyter
- Access Jupyter Notebook:
  ```sh
  Open a browser and go to http://localhost:8888


## ⚡ Using a Supercomputer (HPC)
(Complete this section after setting up the local environment.)

### 🏗 Step 1: Access the HPC System

- 🔑 Login via SSH:
  ```sh
  ssh username@hpc-server-address
- 📦 Load Required Modules:
  - module load python/3.13
  - module load jupyter
- 📡 Start a Jupyter Session on HPC:
  - jupyter notebook --no-browser --port=8888
- 🌍 Access from Local Machine:
  - ssh -N -L 8888:localhost:8888 username@hpc-server-address

## 🏛️ Legacy Content

This wiki ensures step-by-step documentation for users to follow easily, including setup, troubleshooting, and execution.

