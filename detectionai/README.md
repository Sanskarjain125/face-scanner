# Face Scanner website

This project is a website-first face scanner. On Vercel, the camera and face matching run in the visitor's browser; live camera frames are not posted to the Flask/Vercel server. The small Flask app serves the page, enrolled reference images, and profile data.

The native Python scanner and `android_app/` are retained only as local-development material. They are not needed for the website or Vercel deployment.

## What you need

- Python 3.10 or 3.11 (64-bit is recommended).
- A working webcam and a current Chrome, Edge, or Firefox browser.
- Five or six clear reference photos of each person you want to recognise. This requested deployment includes the enrolled face references and a separate object image catalogue.

## First-time installation

Open PowerShell in this `face_scanner` folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.local.txt
python run.py
```

After the dependencies are installed, the only command needed on later runs is:

```powershell
.\.venv\Scripts\python.exe run.py
```

The launcher starts Flask and opens your default browser automatically. Leave the PowerShell window open while using the app; press `Ctrl+C` there to stop it.

## Local Android APK (optional, not needed for the website)

The companion Android project is in `android_app/`. Its debug APK is built at `android_app/app/build/outputs/apk/debug/FaceScanner.apk`. Install it on a phone connected to the same Wi-Fi as this PC. The app uses the phone's native front camera and sends the captured scan only to `http://192.168.29.163:5000/`; it does not depend on a browser camera at that LAN address. If your PC’s Wi-Fi IP changes, replace the URL in the APK's top address field with the current `http://<PC-IP>:5000/`, then tap **Connect**.

Keep `python run.py` running on the PC while using the APK. The recognition backend stays on the PC because the requested Flask + dlib `face_recognition` stack is desktop Python software; the APK securely acts as the mobile camera client. If Windows Firewall asks about Python on a private network, allow it only on your private home/office network so the phone can connect.

For the desktop browser on the PC itself, open `http://127.0.0.1:5000/`. Do not use the `192.168.x.x` address in a desktop browser when you need its webcam: browsers allow camera access on local `127.0.0.1`, but block it on a plain-HTTP LAN address.

### Windows dependency note

The requirements use `dlib-bin==19.24.6`, a pinned prebuilt Windows wheel that provides the `dlib` module used by `face-recognition`. This avoids a CMake and Visual Studio compiler setup on standard 64-bit Python 3.10/3.11 Windows installations. The recognition code remains the requested dlib-based `face_recognition` stack.

## Enrol reference faces

1. For a new clone, copy `details.example.json` to `details.json`, then enter your own local profile details. This repository currently versions `details.json` and `known_faces/` because the requested hosted demo needs them; treat those files as personal and biometric data.
2. Put your images in `known_faces/`. Each image must contain exactly one, clear, forward-facing face. Good lighting and slight angle/expression changes across photos improve reliability.
3. Name all photos for a person with the same id followed by a sequence number. For example:

   ```text
   known_faces/person1_1.jpg
   known_faces/person1_2.jpg
   known_faces/person1_3.jpg
   known_faces/person1_4.jpg
   known_faces/person1_5.jpg
   known_faces/person1_6.jpg
   ```

4. Edit `details.json` so its top-level key exactly matches that id. The value may contain any fields you want displayed:

   ```json
   {
     "person1": {
       "name": "Avery Patel",
       "age": "28",
       "department": "Research",
       "employee_id": "EMP-1042"
     }
   }
   ```

5. Restart `python run.py` after changing either reference images or `details.json`. Reference encodings are intentionally loaded once on startup for speed and consistency.

## Object image catalogue

The supplied object references live in `object_images/`, grouped by category folder: `Accordion`, `Adapter`, `Air_Conditioner`, `Office_Chair`, `Bottle`, `Chair`, `Charger`, `Charging_Point`, `Headphones`, and `Laptop`. The last six were built from the flat photos that used to sit in `bottle1/` (each object's photo(s) moved into its own named subfolder here).

Each folder is matched, case-insensitively, against its keyword in `physical_object_dictionary_5000.json` (via that entry's `image_folder`/`slug` field). On the **Object Dictionary** tab, searching or clicking a keyword that has a matching folder shows its keyword, description, purpose ("Why Use It"), *and* the supplied photos together in the detail card. A **"📷 Photographed objects"** chip row lets you jump straight to any keyword that currently has real photos.

To add a new photographed object: drop one or more images into a new `object_images/<Category_Name>/` folder, restart `python run.py`, and make sure a dictionary entry exists (or add one from the "Add New Custom Word" form) whose `slug`/`image_folder` matches the folder name in lower case (e.g. folder `Ear_Buds` ↔ keyword slug `ear_buds`).

## Use the scanner

Click **Start Camera**, approve the browser's camera permission, and look into the preview with only one face visible. The app scans about every 2.5 seconds; **Scan Face** triggers a scan immediately. When a face matches, scanning pauses and the recognised profile remains visible. Click **Restart Scanning** only when you want to clear that result and scan again. A non-match displays **Face not recognized**.

The match threshold is configured by `FACE_MATCH_THRESHOLD = 0.55` near the top of `app.py`. Lower it (for example, `0.50`) to reduce false matches; raise it carefully only if valid matches are repeatedly missed.

## Troubleshooting

- **No reference faces are loaded:** add valid JPG, JPEG, PNG, or WEBP photos to `known_faces/`, with exactly one face per image, then restart.
- **No face detected:** improve lighting, face the camera, and move closer.
- **Multiple faces detected:** keep only one person in the frame.
- **Low-light message:** increase front lighting; very dark frames are rejected before recognition.
- **Camera permission denied:** use the browser’s site permissions to allow the camera for `127.0.0.1`, then reload the page.
- **Port 5000 is in use:** stop the other local server using that port, then run the launcher again.

## Privacy and limitations

Face recognition is probabilistic. Use it only with the knowledge and permission of the people enrolled, and do not use it as the sole basis for high-impact decisions. Keep `known_faces/` and `details.json` private because they contain biometric and personal information.

## Vercel deployment

The repository includes `vercel.json` and `.python-version` for a lightweight Flask Function. Vercel imports `app.py` directly; it does not run `run.py`. The deployed browser downloads the recognition model, encodes the enrolled reference images, and matches the camera feed locally in the browser. This works without Vercel-native dlib packages.

### Result-email setup

Each recognised profile card includes a **Send result by email** button. The server sends only to the fixed enrolled-profile addresses configured in `app.py`; the browser cannot choose a recipient or access a mail credential. To enable delivery on Vercel, configure these production environment variables and redeploy:

- `RESEND_API_KEY` — an API key created in your Resend account.
- `RESEND_FROM_EMAIL` — a verified Resend sender, for example `Face Scanner <scanner@your-domain.com>`.

Without these variables, the button safely explains that email delivery is not configured and no scan data is sent.

For local PowerShell testing, set the same variables in the terminal before starting Flask:

```powershell
$env:RESEND_API_KEY = "re_your_api_key"
$env:RESEND_FROM_EMAIL = "Face Scanner <scanner@your-verified-domain.com>"
C:/Users/Hp/AppData/Local/Programs/Python/Python314/python.exe -m flask --app app run --host 127.0.0.1 --port 5000
```

Sanskar Jain's result is sent to `sanskarjain@appicsoftwares.in`; the sender address must be verified in Resend. Never commit the API key to this repository.

`known_faces/` and `details.json` are currently versioned and therefore will be deployed with the site. Confirm that every enrolled person has explicitly agreed to this before making the repository or Vercel project public. Use a private GitHub repository and Vercel access controls if the data is not intended for public access.
