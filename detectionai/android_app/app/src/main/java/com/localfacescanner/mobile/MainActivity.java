package com.localfacescanner.mobile;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.hardware.Camera;
import android.os.Bundle;
import android.util.Base64;
import android.view.Gravity;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Local mobile client. The native camera avoids browser HTTPS restrictions on LAN IPs. */
@SuppressWarnings("deprecation")
public class MainActivity extends Activity implements SurfaceHolder.Callback {
    private static final int CAMERA_PERMISSION_REQUEST = 45;
    private static final String PREFS_NAME = "face_scanner";
    private static final String SERVER_URL_KEY = "server_url";
    private static final String DEFAULT_SERVER_URL = "http://192.168.29.163:5000/";

    private EditText serverUrlInput;
    private TextView statusView;
    private TextView resultView;
    private SurfaceHolder previewHolder;
    private Button scanButton;
    private Button connectButton;
    private Camera camera;
    private ExecutorService networkExecutor;
    private boolean serviceReady;
    private boolean isCapturing;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        networkExecutor = Executors.newSingleThreadExecutor();
        buildLayout();
        connectToServer();
    }

    private void buildLayout() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(0xff0b1020);
        int padding = dp(14);
        root.setPadding(padding, padding, padding, padding);
        root.setOnApplyWindowInsetsListener((view, insets) -> {
            root.setPadding(padding + insets.getSystemWindowInsetLeft(), padding + insets.getSystemWindowInsetTop(), padding + insets.getSystemWindowInsetRight(), padding + insets.getSystemWindowInsetBottom());
            return insets;
        });

        TextView title = text("Face Scanner", 28, 0xffeef4ff);
        title.setGravity(Gravity.CENTER);
        root.addView(title, matchWrap());
        TextView hint = text("Your phone camera sends scans only to the Face Scanner service running on your PC. Keep both devices on the same Wi-Fi.", 13, 0xffa9b9d8);
        hint.setGravity(Gravity.CENTER);
        hint.setPadding(0, dp(8), 0, dp(8));
        root.addView(hint, matchWrap());

        LinearLayout serverRow = new LinearLayout(this);
        serverRow.setGravity(Gravity.CENTER_VERTICAL);
        serverUrlInput = new EditText(this);
        serverUrlInput.setSingleLine(true);
        serverUrlInput.setText(getSharedPreferences(PREFS_NAME, MODE_PRIVATE).getString(SERVER_URL_KEY, DEFAULT_SERVER_URL));
        serverUrlInput.setHint("http://PC-IP:5000/");
        serverUrlInput.setTextColor(0xffffffff);
        serverUrlInput.setHintTextColor(0xffa9b9d8);
        serverRow.addView(serverUrlInput, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1));
        connectButton = new Button(this);
        connectButton.setText("Connect");
        connectButton.setOnClickListener(view -> connectToServer());
        serverRow.addView(connectButton, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT));
        root.addView(serverRow, matchWrap());

        statusView = text("Checking scanner service…", 13, 0xffc7d5ee);
        statusView.setPadding(0, 0, 0, dp(10));
        root.addView(statusView, matchWrap());

        SurfaceView preview = new SurfaceView(this);
        preview.setBackgroundColor(0xff050914);
        previewHolder = preview.getHolder();
        previewHolder.addCallback(this);
        root.addView(preview, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1));

        scanButton = new Button(this);
        scanButton.setText("Scan Face");
        scanButton.setEnabled(false);
        scanButton.setOnClickListener(view -> captureAndScan());
        LinearLayout.LayoutParams scanParams = matchWrap();
        scanParams.topMargin = dp(10);
        root.addView(scanButton, scanParams);

        resultView = text("", 16, 0xffdce8ff);
        resultView.setPadding(dp(8), dp(12), dp(8), 0);
        root.addView(resultView, matchWrap());
        setContentView(root);
    }

    private LinearLayout.LayoutParams matchWrap() {
        return new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
    }

    private TextView text(String value, int size, int color) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextSize(size);
        view.setTextColor(color);
        return view;
    }

    private void connectToServer() {
        final String baseUrl = normalizedServerUrl();
        if (baseUrl == null) return;
        serviceReady = false;
        scanButton.setEnabled(false);
        connectButton.setEnabled(false);
        setStatus("Connecting to " + baseUrl, 0xffc7d5ee);
        networkExecutor.execute(() -> {
            try {
                HttpURLConnection connection = openConnection(endpoint(baseUrl, "/api/status"), "GET");
                int code = connection.getResponseCode();
                String body = readText(code < 400 ? connection.getInputStream() : connection.getErrorStream());
                connection.disconnect();
                if (code != 200) throw new IllegalStateException("Server returned HTTP " + code);
                JSONObject status = new JSONObject(body);
                boolean ready = status.optBoolean("ready", false);
                int profiles = status.optInt("reference_count", 0);
                runOnUiThread(() -> {
                    serviceReady = ready;
                    scanButton.setEnabled(ready && camera != null && !isCapturing);
                    setStatus(ready ? profiles + " enrolled profile(s) loaded. Camera is ready." : "Server connected, but no reference faces are loaded.", ready ? 0xff91e7bb : 0xffffd483);
                    connectButton.setEnabled(true);
                });
            } catch (Exception error) {
                runOnUiThread(() -> {
                    setStatus("Cannot reach PC server. Start python run.py, use same Wi-Fi, and check the address. (" + error.getMessage() + ")", 0xffffb4b4);
                    connectButton.setEnabled(true);
                });
            }
        });
    }

    private String normalizedServerUrl() {
        String value = serverUrlInput.getText().toString().trim();
        if (value.isEmpty()) { serverUrlInput.setError("Enter the PC server address"); return null; }
        if (!value.startsWith("http://") && !value.startsWith("https://")) value = "http://" + value;
        if (!value.endsWith("/")) value += "/";
        try { if (new URL(value).getHost().isEmpty()) throw new IllegalArgumentException(); }
        catch (Exception error) { serverUrlInput.setError("Use a URL such as http://192.168.29.163:5000/"); return null; }
        serverUrlInput.setText(value);
        getSharedPreferences(PREFS_NAME, MODE_PRIVATE).edit().putString(SERVER_URL_KEY, value).apply();
        return value;
    }

    private String endpoint(String baseUrl, String path) throws Exception {
        URL base = new URL(baseUrl);
        return new URL(base.getProtocol(), base.getHost(), base.getPort(), path).toString();
    }

    private HttpURLConnection openConnection(String address, String method) throws Exception {
        HttpURLConnection connection = (HttpURLConnection) new URL(address).openConnection();
        connection.setConnectTimeout(8000);
        connection.setReadTimeout(25000);
        connection.setRequestMethod(method);
        return connection;
    }

    @Override public void surfaceCreated(SurfaceHolder holder) { openCameraWhenPermitted(); }
    @Override public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) { }
    @Override public void surfaceDestroyed(SurfaceHolder holder) { releaseCamera(); }

    private void openCameraWhenPermitted() {
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) openCamera();
        else requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST);
    }

    private void openCamera() {
        if (camera != null || previewHolder == null) return;
        try {
            int frontCamera = findFrontCamera();
            camera = Camera.open(frontCamera);
            camera.setDisplayOrientation(90);
            camera.setPreviewDisplay(previewHolder);
            camera.startPreview();
            scanButton.setEnabled(serviceReady);
        } catch (Exception error) {
            releaseCamera();
            setStatus("Could not start phone camera: " + error.getMessage(), 0xffffb4b4);
        }
    }

    private int findFrontCamera() {
        Camera.CameraInfo info = new Camera.CameraInfo();
        for (int index = 0; index < Camera.getNumberOfCameras(); index++) {
            Camera.getCameraInfo(index, info);
            if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) return index;
        }
        return 0;
    }

    private void captureAndScan() {
        if (!serviceReady || camera == null || isCapturing) return;
        isCapturing = true;
        scanButton.setEnabled(false);
        setStatus("Capturing and scanning face…", 0xffc7d5ee);
        try {
            camera.takePicture(null, null, (jpeg, activeCamera) -> {
                try { activeCamera.startPreview(); } catch (Exception ignored) { }
                String baseUrl = normalizedServerUrl();
                if (baseUrl != null) networkExecutor.execute(() -> uploadAndScan(baseUrl, jpeg));
            });
        } catch (Exception error) {
            isCapturing = false;
            scanButton.setEnabled(serviceReady);
            setStatus("Camera capture failed: " + error.getMessage(), 0xffffb4b4);
        }
    }

    private void uploadAndScan(String baseUrl, byte[] jpeg) {
        try {
            JSONObject payload = new JSONObject();
            payload.put("image", "data:image/jpeg;base64," + Base64.encodeToString(jpeg, Base64.NO_WRAP));
            HttpURLConnection connection = openConnection(endpoint(baseUrl, "/scan"), "POST");
            connection.setDoOutput(true);
            connection.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
            try (OutputStream output = connection.getOutputStream()) { output.write(payload.toString().getBytes(StandardCharsets.UTF_8)); }
            int code = connection.getResponseCode();
            String body = readText(code < 400 ? connection.getInputStream() : connection.getErrorStream());
            connection.disconnect();
            JSONObject result = new JSONObject(body);
            runOnUiThread(() -> showResult(result, code));
        } catch (Exception error) {
            runOnUiThread(() -> {
                isCapturing = false;
                scanButton.setEnabled(serviceReady && camera != null);
                setStatus("Scan failed. Check that the PC server is running. (" + error.getMessage() + ")", 0xffffb4b4);
            });
        }
    }

    private void showResult(JSONObject result, int code) {
        isCapturing = false;
        scanButton.setEnabled(serviceReady && camera != null);
        String message = result.optString("message", "Scan complete.");
        if (code >= 400) { resultView.setText(""); setStatus(message, 0xffffb4b4); return; }
        if (result.optBoolean("match", false)) {
            String name = result.optString("name", "Recognized face");
            String confidence = result.optString("confidence", "");
            String clothes = result.optString("clothes_colour", "");
            resultView.setText("✓ Recognized: " + name + (confidence.isEmpty() ? "" : "\nConfidence: " + confidence + "%") + (clothes.isEmpty() ? "" : "\nClothes colour: " + clothes));
            resultView.setTextColor(0xff91e7bb);
            setStatus("Face recognized. Tap Scan Face to scan again.", 0xff91e7bb);
        } else {
            resultView.setText("! " + message);
            resultView.setTextColor(0xffffc3ca);
            setStatus(message, 0xffffd483);
        }
    }

    private String readText(InputStream stream) throws Exception {
        if (stream == null) return "";
        try (InputStream input = stream; ByteArrayOutputStream bytes = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[4096]; int read;
            while ((read = input.read(buffer)) != -1) bytes.write(buffer, 0, read);
            return bytes.toString("UTF-8");
        }
    }

    private void setStatus(String message, int color) {
        runOnUiThread(() -> { if (statusView != null) { statusView.setText(message); statusView.setTextColor(color); } });
    }

    private int dp(int value) { return Math.round(value * getResources().getDisplayMetrics().density); }

    private void releaseCamera() {
        if (camera != null) { try { camera.stopPreview(); } catch (Exception ignored) { } camera.release(); camera = null; }
    }

    @Override public void onRequestPermissionsResult(int code, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(code, permissions, results);
        if (code == CAMERA_PERMISSION_REQUEST) {
            if (results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) openCamera();
            else setStatus("Camera permission is required to scan faces. Allow it in Android app settings.", 0xffffb4b4);
        }
    }

    @Override protected void onDestroy() {
        releaseCamera();
        if (networkExecutor != null) networkExecutor.shutdownNow();
        super.onDestroy();
    }
}
