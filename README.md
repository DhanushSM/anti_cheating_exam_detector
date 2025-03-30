# Anti-Cheating Exam Detector

## About the Project
The Anti-Cheating Exam Detector is a Python-based application designed to detect and prevent cheating during online examinations. The system leverages computer vision techniques to monitor the test-taker's environment and detect suspicious activities. This innovative approach aims to ensure the integrity and fairness of online exams by providing real-time monitoring and alerting capabilities.

## Features
- **Real-time Monitoring:** Continuously monitors the test-taker's environment using the camera feed.
- **Cheating Detection:** Detects suspicious activities such as multiple faces, unauthorized devices, and unusual movements.
- **Alerts and Notifications:** Sends real-time alerts and notifications to the proctor in case of detected cheating activities.
- **User-Friendly Interface:** Provides a simple and intuitive interface for both test-takers and proctors.

## How It Works
1. **Camera Input:** The system captures video input from the test-taker's camera.
2. **Activity Detection:** The captured video frames are processed using computer vision algorithms to detect suspicious activities.
3. **Alert Generation:** If any cheating activities are detected, the system generates real-time alerts and notifications.
4. **Proctor Interface:** The proctor can view the alerts and take appropriate actions to ensure the integrity of the exam.

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/DhanushSM/anti_cheating_exam_detector.git
    ```
2. Navigate to the project directory:
    ```bash
    cd anti_cheating_exam_detector
    ```
3. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```
4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5. Run the application:
    ```bash
    python main.py
    ```

## Usage
- Ensure your camera is connected and properly configured.
- Run the application using the provided command.
- Follow the on-screen instructions to calibrate the detection system.
- Start the exam and monitor the test-taker's environment in real-time.

## Contributing
Contributions are welcome! To contribute to this project, follow these steps:

1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature-name
    ```
3. Make your changes and commit them:
    ```bash
    git commit -m 'Add some feature'
    ```
4. Push to the branch:
    ```bash
    git push origin feature-name
    ```
5. Open a pull request to the `main` branch.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact Information
- **Repository Owner:** [DhanushSM](https://github.com/DhanushSM)
- **Email:** [your-email@example.com]
