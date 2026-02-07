# SSH X11 Forwarding Setup Guide

## ✅ Configuration Complete

Your SSH config has been updated to enable X11 forwarding for the `hendrix` server.

---

## 📝 What Was Added

```ssh
Host hendrix
    HostName hendrix.ad.elucid.biz
    User jin.park
    IdentityFile  ~/.ssh/jin_park
    ForwardX11 yes              # Enable X11 forwarding
    ForwardX11Trusted yes       # Allow full X11 access (trusted)
    Compression yes             # Enable compression for better performance
```

### Configuration Details:

- **ForwardX11 yes** - Enables X11 forwarding
- **ForwardX11Trusted yes** - Allows trusted X11 forwarding (no security restrictions)
- **Compression yes** - Compresses data for faster transmission over network

---

## 🚀 How to Use

### Method 1: Using SSH Config (Recommended)

Simply connect to the server:

```bash
ssh hendrix
```

The X11 forwarding is **automatically enabled** because of the config!

### Method 2: Manual X11 Forwarding (Without Config)

If you want to override or test:

```bash
# Standard X11 forwarding
ssh -X hendrix

# Trusted X11 forwarding (faster, recommended for internal networks)
ssh -Y hendrix

# With compression for better performance
ssh -Y -C hendrix
```

---

## 🧪 Testing X11 Forwarding

### Step 1: Connect to Server

```bash
ssh hendrix
```

### Step 2: Verify X11 Display is Set

Once connected, check if DISPLAY is set:

```bash
echo $DISPLAY
# Should output something like: localhost:10.0 or localhost:11.0
```

If DISPLAY is empty, X11 forwarding is not working.

### Step 3: Test with Simple X11 App

```bash
# Test with xeyes (should display on your local laptop)
xeyes

# Or test with xclock
xclock
```

If a window appears on your **local Ubuntu laptop**, X11 forwarding is working! ✅

---

## 🎨 Running the Medical Imaging Framework GUI

### On the Server (hendrix)

```bash
# Connect with X11 forwarding
ssh hendrix

# Navigate to the project
cd ~/Codes/Node-MedicalImaging-Framework

# Launch the GUI
python -m medical_imaging_framework.gui.editor

# Or launch example GUI
python examples/medical_segmentation_pipeline/launch_gui.py
```

The GUI window will appear on your **local Ubuntu laptop**! 🎉

---

## 🔧 Troubleshooting

### Issue 1: "DISPLAY is not set"

**Solution on your local laptop (before connecting):**

```bash
# Check X11 is running
echo $DISPLAY
# Should show :0 or similar

# If empty, make sure you're in a graphical session
# Restart your terminal or log out and back in
```

### Issue 2: "Can't open display"

**Check on server:**
```bash
echo $DISPLAY
# If empty, X11 forwarding failed
```

**Solution:**
```bash
# Reconnect with verbose mode to see errors
ssh -v hendrix 2>&1 | grep -i x11
```

**Common causes:**
- X11 forwarding disabled on server
- xauth not installed on server
- X11 not running on local laptop

### Issue 3: X11 Forwarding Blocked by Server

**On the server (requires sudo):**

Check if X11 forwarding is allowed:
```bash
grep X11Forwarding /etc/ssh/sshd_config
```

Should show:
```
X11Forwarding yes
```

If not, ask your system administrator to enable it.

### Issue 4: Performance Issues (Slow GUI)

**Solution - Enable compression:**

```bash
# Already configured in your SSH config!
# But you can also force maximum compression:
ssh -Y -C -o CompressionLevel=9 hendrix
```

**Or use SSH with lower quality:**
```bash
# Disable compression for faster but lower quality
ssh -Y -o Compression=no hendrix
```

---

## 🔐 Security Notes

### ForwardX11 vs ForwardX11Trusted

**ForwardX11 (ssh -X):**
- ✅ More secure
- ⚠️ Some X11 features may not work
- 🐌 May have security restrictions

**ForwardX11Trusted (ssh -Y):**
- ✅ Full X11 functionality
- ✅ Better performance
- ⚠️ Less secure (but fine for trusted internal networks)

**Your config uses ForwardX11Trusted** - appropriate for internal company networks.

---

## 📊 Performance Tips

### For Better GUI Performance:

1. **Use compression** (already enabled in config)
   ```bash
   Compression yes
   ```

2. **Close unnecessary applications** on the server to reduce X11 traffic

3. **Use faster network connection** if possible (wired > WiFi)

4. **Consider SSH connection multiplexing** (reuse connections):
   ```ssh
   ControlMaster auto
   ControlPath ~/.ssh/control-%h
   ControlPersist 10m
   ```

---

## 🎯 Quick Reference

### Connect to Server with GUI Support
```bash
ssh hendrix
# X11 forwarding is automatic!
```

### Test X11 is Working
```bash
xeyes
# or
xclock
```

### Launch Medical Imaging Framework GUI
```bash
cd ~/Codes/Node-MedicalImaging-Framework
python -m medical_imaging_framework.gui.editor
```

### Check X11 Display
```bash
echo $DISPLAY
# Should show: localhost:10.0 or similar
```

### Debug X11 Issues
```bash
ssh -v hendrix 2>&1 | grep -i x11
```

---

## ✅ Checklist

Before running GUI remotely:

**On Local Laptop:**
- [ ] Running Ubuntu desktop (GUI environment)
- [ ] X11 server running (check: `echo $DISPLAY`)
- [ ] SSH client installed
- [ ] SSH config updated (done! ✅)

**On Remote Server (hendrix):**
- [ ] X11Forwarding enabled in /etc/ssh/sshd_config
- [ ] xauth installed (`which xauth`)
- [ ] Project code deployed

**Test Connection:**
- [ ] `ssh hendrix` connects successfully
- [ ] `echo $DISPLAY` shows something like "localhost:10.0"
- [ ] `xeyes` displays a window on local laptop

---

## 🌐 Alternative: VNC (If X11 is Slow)

If X11 forwarding is too slow, consider VNC:

**On Server:**
```bash
vncserver :1 -geometry 1920x1080
```

**On Local Laptop:**
```bash
ssh -L 5901:localhost:5901 hendrix
vncviewer localhost:5901
```

---

## 📚 Additional Resources

- **SSH X11 Forwarding:** https://www.ssh.com/academy/ssh/x11-forwarding
- **Ubuntu X11 Guide:** https://help.ubuntu.com/community/SSH/OpenSSH/PortForwarding
- **PyQt5 over SSH:** Works seamlessly with X11 forwarding!

---

**Your SSH config is now configured for X11 forwarding!** 🎉

Just run `ssh hendrix` and start using GUI applications remotely!
