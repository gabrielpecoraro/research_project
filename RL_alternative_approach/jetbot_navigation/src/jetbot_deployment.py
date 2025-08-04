#!/usr/bin/env python3
"""
Deploy JetBot Navigation to actual JetBot hardware
"""

import os
import sys
import subprocess
import paramiko
import getpass


def deploy_to_jetbot():
    """Deploy to JetBot via SSH"""

    # Get JetBot connection details
    jetbot_ip = input("Enter JetBot IP address: ")
    jetbot_user = input("Enter JetBot username (default: jetbot): ") or "jetbot"
    jetbot_password = getpass.getpass("Enter JetBot password: ")

    try:
        # Create SSH connection
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(jetbot_ip, username=jetbot_user, password=jetbot_password)

        # Create SFTP connection for file transfer
        sftp = ssh.open_sftp()

        print("Connected to JetBot!")

        # Transfer files
        print("Transferring files...")
        local_dir = "."
        remote_dir = "/home/jetbot/jetbot_navigation"

        # Create remote directory
        ssh.exec_command(f"mkdir -p {remote_dir}")

        # Transfer all files
        transfer_directory(sftp, local_dir, remote_dir)

        print("Files transferred successfully!")

        # Run setup on JetBot
        print("Running setup on JetBot...")

        commands = [
            f"cd {remote_dir}",
            "chmod +x setup.sh",
            "./setup.sh",
            "python3 deploy.py",
        ]

        for cmd in commands:
            stdin, stdout, stderr = ssh.exec_command(cmd)
            output = stdout.read().decode()
            error = stderr.read().decode()

            if output:
                print(f"OUTPUT: {output}")
            if error:
                print(f"ERROR: {error}")

        print("Deployment complete!")
        print(f"SSH to JetBot: ssh {jetbot_user}@{jetbot_ip}")
        print(f"Run: cd {remote_dir} && python3 examples/basic_navigation.py")

    except Exception as e:
        print(f"Deployment failed: {e}")
        return False
    finally:
        try:
            sftp.close()
            ssh.close()
        except:
            pass

    return True


def transfer_directory(sftp, local_dir, remote_dir):
    """Recursively transfer directory"""
    for item in os.listdir(local_dir):
        local_path = os.path.join(local_dir, item)
        remote_path = f"{remote_dir}/{item}"

        if os.path.isdir(local_path):
            # Create remote directory
            try:
                sftp.mkdir(remote_path)
            except:
                pass  # Directory might already exist

            # Recursively transfer subdirectory
            transfer_directory(sftp, local_path, remote_path)
        else:
            # Transfer file
            print(f"Transferring {local_path} -> {remote_path}")
            sftp.put(local_path, remote_path)


if __name__ == "__main__":
    # Install paramiko for SSH if not available
    try:
        import paramiko
    except ImportError:
        print("Installing paramiko for SSH deployment...")
        subprocess.run([sys.executable, "-m", "pip", "install", "paramiko"])
        import paramiko

    deploy_to_jetbot()
