# utils/automation_api.py
import subprocess

def trigger_automation(mode='once'):
    try:
        if mode=='once':
            result=subprocess.run(['python','../run.py','--mode','once'],capture_output=True,text=True)
        elif mode=='publish':
            result=subprocess.run(['python','../run.py','--mode','publish'],capture_output=True,text=True)
        else:
            return f"Unknown mode: {mode}"

        if result.returncode==0:
            return "Automation executed successfully!"
        else:
            return f"Automation failed: {result.stderr}"
    except Exception as e:
        return f"Error running automation: {str(e)}"
