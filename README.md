# Face Detection Android App

This is the Android version of the Face Detection System, built using Python, Kivy, and OpenCV.

## Prerequisites

- Python 3.7 or higher
- Java Development Kit (JDK) 8 or higher
- Android SDK
- Android NDK
- Buildozer

## Development Setup

1. Install the required Python packages:
```bash
pip install kivy kivymd opencv-python numpy pillow plyer buildozer
```

2. Install Android development prerequisites:
- Install [Android Studio](https://developer.android.com/studio)
- Install Android SDK through Android Studio
- Install Android NDK through Android Studio

3. Set environment variables:
- JAVA_HOME: Path to JDK installation
- ANDROID_HOME: Path to Android SDK installation
- ANDROID_NDK_HOME: Path to Android NDK installation

## Building the App

### On Linux (Recommended)
```bash
# Initialize buildozer
buildozer init

# Build debug APK
buildozer android debug

# Build release APK
buildozer android release
```

### On Windows
It's recommended to use WSL (Windows Subsystem for Linux) or a Linux virtual machine for building the Android APK. However, you can develop and test the app on Windows using:
```bash
python main.py
```

## Features

- Real-time face detection using device camera
- Photo capture and gallery support
- Dark/Light theme support
- Statistics dashboard
- Backup/restore functionality
- Battery-saving mode
- Adjustable resolution settings

## App Structure

- `main.py`: Main application file
- `face_detection.kv`: Kivy UI layout file
- `buildozer.spec`: Build configuration for Android
- `haarcascade_frontalface_default.xml`: Face detection cascade file

## Testing

1. Test on desktop:
```bash
python main.py
```

2. Test on Android device:
- Enable USB debugging on your Android device
- Connect device via USB
- Run: `buildozer android debug deploy run`

## Troubleshooting

1. Camera issues:
   - Make sure camera permissions are granted
   - Check Android Manifest settings in buildozer.spec

2. Build issues:
   - Verify all prerequisites are installed
   - Check environment variables
   - Use a Linux environment for building

3. Performance issues:
   - Adjust resolution settings
   - Enable battery saver mode for longer sessions
