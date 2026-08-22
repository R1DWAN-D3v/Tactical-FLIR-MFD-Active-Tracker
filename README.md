# Tactical-FLIR-MFD-Optical-Flow-Tracker
A real-time military Multi-Function Display (MFD) and Forward-Looking Infrared (FLIR) HUD simulation powered by a hybrid Lucas-Kanade Optical Flow + Thermal Centroid Tracking Engine. Designed to track high-speed, maneuvering targets through extreme contrast shifts without freezing or dropping locks.


[![Python Version](https://img.shields.io/badge/python-3.8%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8.svg?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pillow](https://img.shields.io/badge/Pillow-8.0%2B-000000.svg?style=for-the-badge&logo=python&logoColor=white)](https://python-pillow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

A real-time military Multi-Function Display (MFD) and Forward-Looking Infrared (FLIR) HUD simulation powered by a hybrid **Lucas-Kanade Optical Flow + Thermal Centroid Tracking Engine**. Designed to track high-speed, maneuvering targets through extreme contrast shifts without freezing or dropping locks.

---

## 🎯 Key Features

* **Hybrid Tracking Engine:** Combines sparse Lucas-Kanade optical flow feature vector tracking with dynamic thermal centroid fallback for seamless lock retention.
* **Thermal Sensor Emulation:** Features dual-spectrum modes (Green Phosphor IR & Black/White IR) with dynamic contrast enhancement (CLAHE) and Polarity Inversion (White Hot / Black Hot).
* **ATFLIR HUD & MFD Overlay:** Military-spec reticle with dynamic telemetry readouts, velocity calculation in Knots, pitch/roll indicators, and simulated GPS grid coordinates.
* **Multi-Target Hotspot Detection:** Automatically scans and highlights secondary thermal heat signatures across the Field of View (FOV).
* **Real-time Digital Zoom:** Hardware-accelerated center cropping for 1.0x, 2.0x, and 4.0x magnification.

---

## 📂 Directory Structure

```text
tactical-flir-mfd-tracker/
├── mfd_flir_system.py     # Main application script
├── requirements.txt       # Project dependencies
├── assets/                # Screenshots and documentation media
├── LICENSE                # MIT License
└── README.md              # Repository documentation
🎮 Controls & Hotkeys
Key Input	Command	Action Description
L / Space	Lock Target	Engages hybrid optical flow lock at reticle crosshair
W A S D	Slew Reticle	Manually drives target reticle (disengages active lock)
T	Toggle Mode	Switches sensor pipeline between IR-GRN and IR-B&W
P	Toggle Polarity	Swaps signal polarity between WHT (White-Hot) & BLK (Black-Hot)
Z	Cycle Zoom	Steps digital zoom through Z1.0, Z2.0, and Z4.0
Q / Esc	Exit	Terminates video capture stream and closes interface window
🚀 Installation & Setup
Prerequisites
Python 3.8 or higher installed on your system.

A working web camera or video capture device.

Step-by-Step Instructions
Clone the Repository

Bash
git clone [https://github.com/your-username/tactical-flir-mfd-tracker.git](https://github.com/your-username/tactical-flir-mfd-tracker.git)
cd tactical-flir-mfd-tracker
Create a Virtual Environment

Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate
Install Dependencies

Bash
pip install -r requirements.txt
Run the Application

Bash
python mfd_flir_system.py
⚙️ Technical Architecture
Plaintext
[ Camera Feed ] ──► [ Digital Zoom Crop ] ──► [ CLAHE Contrast Enhancement ]
                                                       │
                                                       ▼
[ Reticle / HUD ] ◄── [ LK Flow + Centroid ] ◄── [ Sensor LUT / Colorization ]
Preprocessing: Stream frames are center-cropped according to the active zoom factor and passed through a Contrast Limited Adaptive Histogram Equalization (CLAHE) filter to maximize infrared edge isolation.

Feature Extraction: Upon pressing lock (L), Shi-Tomasi corner detection extracts high-rigidity feature keypoints inside the primary bounding box.

Tracking Matrix: Pyramidal Lucas-Kanade optical flow calculates point translation across sequential frames. If feature density degrades, the engine defaults to a local spatial moment thermal intensity centroid until re-seeding occurs.

🛠️ Troubleshooting
Black/Frozen Window: Ensure your default webcam index (0) is not currently occupied by another video conferencing application (Zoom, Teams, OBS).

Missing Fonts: The application automatically attempts to fall back to default cross-platform monospaced fonts (consola.ttf, cour.ttf, msgothic.ttc) if custom tactical typefaces are absent.

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for bug reports, performance enhancements, or UI adjustments.

Fork the Project

Create your Feature Branch (git checkout -b feature/AdvancedTelemetry)

Commit your Changes (git commit -m 'Add advanced telemetry readout')

Push to the Branch (git push origin feature/AdvancedTelemetry)

Open a Pull Request

📄 License
Distributed under the MIT License. See LICENSE for more information.
