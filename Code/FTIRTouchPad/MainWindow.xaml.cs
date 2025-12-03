using System.Drawing;
using System.Windows;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using System.IO;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

// ReSharper disable once IdentifierTypo
namespace FTIRTouchPad;

public partial class MainWindow {
    private VideoCapture? _capture;
    private DispatcherTimer? _timer; // voor framerate
    private const int Framerate = 30;

    public MainWindow() {
        InitializeComponent();
        StartCamera();
    }

    private void StartCamera() {
        try {
            _capture = new VideoCapture(1, VideoCapture.API.DShow);
            _capture.ImageGrabbed += ProcessFrame;

            _timer = new DispatcherTimer {
                Interval = TimeSpan.FromMilliseconds(Framerate)
            };
            _timer.Tick += (_, _) => _capture.Grab();
            _timer.Start();
        }
        catch (Exception ex) {
            MessageBox.Show("Unable to access webcam: " + ex.Message);
        }
    }

    private void ProcessFrame(object? sender, EventArgs e) {
        if (_capture is not
            { IsOpened: true }) // pattern matching kijk zowel naar IsOpened als naar of _capture niet null is
            return;

        using var frame = new Mat();
        _capture.Retrieve(frame);
        if (frame.IsEmpty)
            return;

        // PIPELINE stap 1: greyscale
        using var gray = new Mat();
        CvInvoke.CvtColor(frame, gray, ColorConversion.Bgr2Gray);

        // PIPELINE stap 2: gaussian blur
        // Blurren voor de threshold zorgt er voor dat er geen zeer kleine areas (en noise) in de threshold komen
        // maakt het smoother dan enkel een Morphological opening na de threshold
        using var blurred = new Mat();
        CvInvoke.GaussianBlur(gray, blurred, new System.Drawing.Size(7, 7), 1.5);

        // PIPELINE stap 3: threshold
        using var thresh = new Mat();
        CvInvoke.Threshold(blurred, thresh,
            ThresholdSlider.Value, 255,
            ThresholdType.Binary);

        // DEBUG: threshold visualisation
        using var debugThresholdView = new Mat();
        CvInvoke.ApplyColorMap(thresh, debugThresholdView, ColorMapType.Jet);

        // PIPELINE stap 4: vertical padding
        // om de randen te ignoren (waar de leds door het glas heen bleeden als licht puntjes)
        var padding = (int)PaddingSlider.Value;
        using var paddedMask = thresh.Clone();

        // boven rand
        CvInvoke.Rectangle(paddedMask,
            new Rectangle(0, 0, paddedMask.Width, padding),
            new MCvScalar(0), -1);
        // onder rand
        CvInvoke.Rectangle(paddedMask,
            new Rectangle(0, paddedMask.Height - padding,
                paddedMask.Width, padding),
            new MCvScalar(0), -1);

        // DEBUG: padding visualisation
        using var paddingDebugView = frame.Clone();
        // boven rand
        CvInvoke.Rectangle(paddingDebugView,
            new Rectangle(0, 0, frame.Width, padding),
            new MCvScalar(255, 0, 0, 128), -1); // semi-transparent blue
        // onder rand
        CvInvoke.Rectangle(paddingDebugView,
            new Rectangle(0, frame.Height - padding, frame.Width, padding),
            new MCvScalar(255, 0, 0, 128), -1);

        // PIPELINE stap 5: Morphological opening
        using var kernel = CvInvoke.GetStructuringElement(
            MorphShapes.Rectangle, new System.Drawing.Size(10, 10), new System.Drawing.Point(-1, -1));

        using var cleaned = new Mat();
        CvInvoke.MorphologyEx(
            paddedMask, cleaned,
            MorphOp.Open,
            kernel,
            new System.Drawing.Point(-1, -1),
            1,
            BorderType.Default,
            default
        );

        // PIPELINE stap 6: find contours
        var contours = new VectorOfVectorOfPoint();
        CvInvoke.FindContours(
            cleaned, contours, null,
            RetrType.External,
            ChainApproxMethod.ChainApproxSimple);

        // PIPELINE stap 7: teken ellipses
        var displayMat = frame.Clone();

        for (var i = 0; i < contours.Size; i++) {
            var contour = contours[i];
            if (contour.Size < 5) continue;

            var ellipse = CvInvoke.FitEllipse(contour);

            // outline ellipse
            CvInvoke.Ellipse(displayMat, ellipse,
                new MCvScalar(0, 0, 255), 2);
            // center point
            CvInvoke.Circle(displayMat,
                System.Drawing.Point.Round(ellipse.Center),
                5,
                new MCvScalar(0, 255, 0), -1);
        }

        //DEBUG VIEWS
        if (MaskCheckBox.IsChecked == true) {
            displayMat = debugThresholdView.Clone();
        }
        else if (PaddingMaskCheckBox.IsChecked == true) {
            displayMat = paddingDebugView.Clone();
        }


        var source = ToBitmapSource(displayMat);
        Dispatcher.Invoke(() => CameraView.Source = source);
    }


    private static BitmapImage ToBitmapSource(Mat image) {
        using var ms = new MemoryStream();
        image.ToBitmap().Save(ms, System.Drawing.Imaging.ImageFormat.Bmp);
        ms.Seek(0, SeekOrigin.Begin);
        var bmp = new BitmapImage();
        bmp.BeginInit();
        bmp.CacheOption = BitmapCacheOption.OnLoad;
        bmp.StreamSource = ms;
        bmp.EndInit();
        bmp.Freeze();
        return bmp;
    }

    protected override void OnClosing(System.ComponentModel.CancelEventArgs e) {
        base.OnClosing(e);

        _timer?.Stop();

        if (_capture == null) return;
        _capture.Dispose();
        _capture = null;
    }
}