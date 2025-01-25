# Spatial Proximity Excel

This tool is designed to perform spatial proximity analysis on Excel spreadsheets. It simplifies geographic data analysis by allowing users to calculate distances and identify relationships between spatial entities directly from their Excel data.

## Features

- **Spatial Proximity Calculation**: Quickly calculate distances between points in your dataset.
- **Geographic Insights**: Generate insights based on spatial relationships in your data.
- **Excel Integration**: Seamlessly works with Excel files for input and output.

Access it [here](https://spatial-proximity-excel.streamlit.app/) or read on to run locally.

![image](https://github.com/user-attachments/assets/4cceca3f-826b-49d7-bda8-4f0cacfb91b7)

## Prerequisites

- Python 3.x
- pip (Python package installer)

## Installation Instructions

### Windows

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/danmaps/spatial-Proximity-Excel.git
   cd spatial-Proximity-Excel
   ```

2. **Set Up a Virtual Environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

### Linux and macOS

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/danmaps/spatial-Proximity-Excel.git
   cd spatial-Proximity-Excel
   ```

2. **Set Up a Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

### Deactivation
To deactivate the virtual environment, use:
- **Windows:** `venv\Scripts\deactivate`
- **Linux/macOS:** `deactivate`

## Usage

Follow the prompts in the Streamlit app to upload your file and perform spatial analysis.

## Troubleshooting

- **Common Issues:** 
  - Ensure that your Excel file is correctly formatted.
  - Verify that all required dependencies are installed.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
