# 🎰 Lotto Powerball Prediction AI

An advanced AI-powered lottery prediction system with a modern web interface built using Gradio. This application analyzes historical lottery data to generate predictions and provides comprehensive statistical analysis tools.

## 🌟 Features

- **🤖 AI Predictions**: Machine learning-powered lottery number predictions
- **⚙️ Custom Training**: Train models with specific date ranges or recent draws
- **📊 Statistical Analysis**: Comprehensive data analysis and visualization
- **🔥 Heatmaps**: Visual frequency analysis of numbers
- **📈 Trend Analysis**: Track number patterns over time
- **🧩 Pattern Recognition**: Analyze winning combination patterns
- **📝 Data Management**: Add new draws and manage historical data
- **🎯 Number Validation**: Check your numbers against AI predictions

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

Choose your operating system:

## 🐧 Linux Setup

```bash
# Clone or download the repository
git clone <repository-url>
cd lotto

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# If you encounter pandas/numpy compatibility issues:
pip uninstall pandas numpy -y
pip install pandas numpy

# Run the application
python mainv2.py
```

## 🪟 Windows Setup

```cmd
# Clone or download the repository
git clone <repository-url>
cd lotto

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# If you encounter pandas/numpy compatibility issues:
pip uninstall pandas numpy -y
pip install pandas numpy

# Run the application
python mainv2.py
```

### PowerShell (Windows)
```powershell
# Clone or download the repository
git clone <repository-url>
cd lotto

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# If you encounter pandas/numpy compatibility issues:
pip uninstall pandas numpy -y; pip install pandas numpy

# Run the application
python mainv2.py
```

## 🍎 macOS Setup

```bash
# Clone or download the repository
git clone <repository-url>
cd lotto

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# If you encounter pandas/numpy compatibility issues:
pip uninstall pandas numpy -y
pip install pandas numpy

# Run the application
python mainv2.py
```

### Using Homebrew (macOS)
```bash
# Install Python if not already installed
brew install python

# Follow the same steps as Linux setup above
```

## 📦 Dependencies

The application requires the following Python packages:

```
gradio>=4.0.0
pandas>=2.0.0
numpy>=1.20.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

## 🎮 Usage

1. **Start the Application**:
   ```bash
   python mainv2.py
   ```

2. **Open Web Interface**:
   - Open your browser and go to: `http://127.0.0.1:7860`

3. **Initialize the System**:
   - Click "Initialize Data & Train Models" to load and process the data

4. **Explore Features**:
   - **🔮 Predictions**: Generate AI predictions for next draw
   - **⚙️ Training Config**: Customize model training parameters
   - **📊 Data & Likelihoods**: View recent draws and probability tables
   - **🔥 Heatmaps**: Visualize number frequency patterns
   - **📈 Trends**: Analyze number trends over time
   - **🧩 Patterns**: Study winning combination patterns
   - **📝 Data Management**: Add new draws and update data

## 📁 Project Structure

```
lotto/
├── mainv2.py                 # Main Gradio application
├── main.py                   # Original Streamlit application
├── requirements.txt          # Python dependencies
├── cleaned_lotto_powerball.csv    # Processed lottery data
├── calculated_lotto_powerball.csv # Statistical calculations
├── powerball/               # Data processing scripts
│   ├── 1_add_split.py      # Data cleaning script
│   ├── 2_add_calc.py       # Statistics calculation script
│   ├── 3_cal_freq.py       # Frequency calculation script
│   └── powerball2009_2025 - Sheet1.csv  # Raw data
└── lotto/                  # Additional lottery data
```

## 🔧 Troubleshooting

### Common Issues

1. **Pandas/NumPy Compatibility Error**:
   ```bash
   pip uninstall pandas numpy -y
   pip install pandas numpy
   ```

2. **Missing Data Files**:
   - Ensure `powerball/powerball2009_2025 - Sheet1.csv` exists
   - Run data management cleanup scripts

3. **Port Already in Use**:
   - Change port in `mainv2.py`: `demo.launch(server_port=7861)`

4. **Permission Errors (Linux/macOS)**:
   ```bash
   chmod +x mainv2.py
   ```

main.py
![image](https://github.com/user-attachments/assets/5405360e-c631-42de-994b-a973bf6679d0)

mainv2.py
![image](https://github.com/user-attachments/assets/1091ec44-37da-4e54-8571-c06b7699a423)

### Data Sources

For the latest lottery results, visit:
- **South African Powerball**: https://za.national-lottery.com/powerball/results/history

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please check local regulations regarding lottery prediction software.

## 🎯 Advanced Usage

### Custom Model Training

1. **Recent Draws Training**:
   - Go to "⚙️ Training Config" tab
   - Select "Recent Draws"
   - Choose number of draws (50-2000)
   - Click "Train Custom Model"

2. **Date Range Training**:
   - Select "Date Range"
   - Enter start and end dates (YYYY-MM-DD)
   - Train model with specific time periods

3. **Model Comparison**:
   - Train multiple models with different parameters
   - Compare prediction accuracy
   - Use "Reset to Full Dataset" to return to default

### Adding New Lottery Draws

1. **Navigate to Data Management**:
   - Go to "📝 Data Management" tab
   - Enter draw information:
     - Date (e.g., "Friday 8 August 2025")
     - 5 main numbers (1-50)
     - PowerBall number (1-20)
     - Jackpot amount
     - Outcome (Roll/Won)

2. **Process Data**:
   - Click "Add Draw & Run Cleanup"
   - Wait for processing to complete
   - Go to Predictions tab and click "Initialize Data & Train Models"

### Performance Optimization

- **Memory Usage**: For large datasets, consider using recent draws training
- **Processing Speed**: Limit training data for faster model updates
- **Accuracy**: Use at least 100 draws for reliable predictions

## 🔍 Technical Details

### Machine Learning Models

- **Algorithm**: Random Forest Classifier
- **Features**: Statistical measures (mean, median, std, range, etc.)
- **Multi-output**: Separate models for main numbers and PowerBall
- **Validation**: Train/test split with cross-validation

### Data Processing Pipeline

1. **Raw Data**: CSV with multiline format
2. **Cleaning**: Split and normalize data format
3. **Feature Engineering**: Calculate statistical features
4. **Model Training**: Train Random Forest models
5. **Prediction**: Generate probability-based predictions

### System Requirements

- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: 500MB free space
- **CPU**: Multi-core processor recommended
- **Network**: Internet connection for Gradio interface

## 🐛 Known Issues

1. **Large Dataset Performance**: Processing 10,000+ draws may be slow
2. **Browser Compatibility**: Best performance on Chrome/Firefox
3. **Mobile Interface**: Limited mobile optimization

## 🔄 Updates and Maintenance

### Regular Maintenance

1. **Update Data**: Add new draws weekly
2. **Retrain Models**: Refresh models with new data
3. **Backup Data**: Keep backups of CSV files
4. **Update Dependencies**: Keep Python packages updated

### Version History

- **v2.0**: Gradio interface with custom training
- **v1.0**: Original Streamlit application

## 📞 Support

For issues and questions:

1. **Check Troubleshooting Section**: Common solutions provided
2. **Review Error Messages**: Application provides detailed error feedback
3. **Data Validation**: Ensure CSV files are properly formatted
4. **Dependencies**: Verify all required packages are installed

## 🎓 Educational Use

This project demonstrates:

- **Machine Learning**: Classification and prediction algorithms
- **Data Science**: Statistical analysis and visualization
- **Web Development**: Modern UI with Gradio framework
- **Data Processing**: CSV handling and feature engineering

Perfect for:
- **Students**: Learning ML and data science concepts
- **Researchers**: Studying lottery patterns and randomness
- **Developers**: Understanding Gradio and web interfaces

## ⚠️ Disclaimer

This application is for entertainment and educational purposes only. Lottery numbers are random, and past results do not guarantee future outcomes. Please gamble responsibly.

**Important**: This software does not guarantee winning lottery numbers. Use at your own discretion and always follow local gambling regulations.
