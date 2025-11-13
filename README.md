# Image-Style-Transfer

## Reports
- [Proposal](https://liveconcordia-my.sharepoint.com/:w:/g/personal/g_marka_live_concordia_ca/EQWC4O9qFrJFviS6shCktSMBSAPh8qxoPMRqU8wkNr8cGw?e=lDrn7V)
- Presentation Slides (Optional)
- [Final Report](https://liveconcordia-my.sharepoint.com/:w:/g/personal/g_marka_live_concordia_ca/EeqOo3BLOP5OqU9KEqV256YBtA555k3RitFEKknKKe1kVw?e=JKO5Dl)

## Environment Setup

### Building the application

1. **Backend setup**
  - cd into `backend` and create a virtual python environment with `python3 -m venv venv`.
  - Activate the environment with `source venv/bin/activate` and install the dependencies with `pip install flask flask-cors torch torchvision pillow`.
  - Run the backend server with `python app.py`.
  - Once finished, close the virtual environment with `deactivate`.

2. **Frontend setup**
  - cd into `frontend` and install the dependencies with `npm install`.
  - Run the frontend with `npm run start`.

3. **Enjoy :)**
  - Upload the image you want to style to Content-Image and the style you want it to take into Style-Image.
  - Play with the parameters as you need and generate your styled image.

### CUDA preparation (Running the notebook)

1. **Confirm GPU compatibility**
  - Run `nvidia-smi` in PowerShell (or use the NVIDIA Control Panel) to confirm that your system has an NVIDIA GPU with at least Compute Capability 5.0.
  - Note the driver version; you will need a driver that supports CUDA 12.x. Update from [NVIDIA's download page](https://www.nvidia.com/Download/index.aspx) if required.

2. **Install the CUDA Toolkit (optional but recommended)**
  - Download the [CUDA 12.1 Toolkit](https://developer.nvidia.com/cuda-12-1-0-download-archive) and choose the installer that matches your OS.
  - During installation, keep the default options so the toolkit binaries (e.g., `nvcc`) are added to your PATH. This step is optional for PyTorch binaries, but it provides useful developer tools.

3. **Update to the matching cuDNN release (if needed for other projects)**
  - For this notebook, cuDNN ships within the PyTorch wheel, so no extra install is required. If you use native CUDA workflows, install cuDNN for CUDA 12.x from the [NVIDIA Developer site](https://developer.nvidia.com/cudnn) and copy the libraries into the CUDA toolkit folder.

4. **Validate the CUDA installation**
  - Reboot if the installer asks. Then run `nvidia-smi` again to confirm the driver is active.
  - If you installed the full toolkit, run `nvcc --version` to verify the compiler is available and reports CUDA 12.1.

1. **Install CUDA capable drivers**
   - Ensure an NVIDIA GPU with CUDA 12.1 compatible drivers is installed. Update the driver from the NVIDIA control panel or [NVIDIA's download page](https://www.nvidia.com/Download/index.aspx) if needed.

2. **Create or update the project environment**
   - From the project root, create the `.conda` environment (skip if it already exists):
     ```powershell
     conda create --prefix (Resolve-Path .\.conda) python=3.12 -y
     ```
   - Activate the environment whenever you work on the project:
     ```powershell
     conda activate (Resolve-Path .\.conda)
     ```

3. **Install GPU-enabled PyTorch stack**
   - Use the official PyTorch wheel index that matches CUDA 12.1:
     ```powershell
     python -m pip install --upgrade pip
     python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```

4. **Install supporting libraries**
   - Matplotlib and Pillow are required for plotting and image I/O:
     ```powershell
     python -m pip install matplotlib pillow
     ```

5. **Verify the installation**
   - Confirm that PyTorch detects CUDA and the expected packages are present:
     ```powershell
     python -c "import torch, matplotlib, PIL; print(f'PyTorch: {torch.__version__} CUDA available: {torch.cuda.is_available()}'); print(f'Matplotlib: {matplotlib.__version__}'); print(f'Pillow: {PIL.__version__}')"
     ```

6. **Launching the notebook**
   - With the environment activated, start Jupyter and open `main.ipynb`:
     ```powershell
     python -m pip install jupyter
     python -m jupyter notebook
     ```
   - Select the `.conda` kernel when prompted.

**Notes**
- Always run `python -m pip ...` while the environment is active so packages install into `.conda`.
- The `.conda\Scripts` directory is not on the system `PATH`; rely on `python -m` invocations to avoid script warnings.
- If you need to reset the environment, remove the `.conda` directory and repeat steps 2-4.
