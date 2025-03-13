# Running the AI Smart Four-Wheel Drive Car Project at Home

This guide will help you run the AI Smart Four-Wheel Drive Car project on your home computer without requiring the actual robot hardware.

## Prerequisites

- Python 3.6 or higher
- A webcam (optional, the application can run with a simulated camera)
- Windows, macOS, or Linux operating system

## Running in Simulation Mode

The project includes a simulation mode that allows you to test the web interface and functionality without the actual robot hardware.

### On Windows

1. Open Command Prompt or PowerShell
2. Navigate to the project directory
3. Run the following command:
   ```
   run.bat --simulation
   ```

### On macOS or Linux

1. Open Terminal
2. Navigate to the project directory
3. Run the following command:
   ```
   ./run.sh --simulation
   ```
   
   If you get a permission error, you may need to make the script executable first:
   ```
   chmod +x run.sh
   ./run.sh --simulation
   ```

## Accessing the Web Interface

Once the application is running, open your web browser and go to:
```
http://localhost:5000
```

You should see the web interface with a simulation mode banner at the top.

## Features Available in Simulation Mode

In simulation mode, you can:

1. Control the virtual robot using the left joystick
2. Control the virtual camera gimbal using the right joystick
3. View a simulated camera feed
4. Test the SLAM functionality if the SLAM module is available

All robot movements and camera controls will be logged to the console instead of being sent to actual hardware.

## Troubleshooting

### "Module not found" errors

If you see errors about missing modules, try running the installation manually:

```
pip install -r requirements.txt
```

### Camera issues

If you have a webcam but the application is not detecting it, you can still use simulation mode to test the interface.

### Port already in use

If port 5000 is already in use on your system, you can modify the `app.py` file to use a different port:

1. Open `app.py` in a text editor
2. Find the line: `socketio.run(app, host='0.0.0.0', port=5000, debug=True)`
3. Change `port=5000` to a different port, e.g., `port=5001`
4. Save the file and run the application again
5. Access the web interface at `http://localhost:5001` (or whatever port you chose)

## Next Steps

Once you've tested the application in simulation mode, you can:

1. Deploy it to a Raspberry Pi with the actual robot hardware
2. Modify the code to add new features
3. Customize the web interface to your liking

Enjoy experimenting with your AI Smart Four-Wheel Drive Car project! 