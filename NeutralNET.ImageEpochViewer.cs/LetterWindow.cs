using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Drawing.Text;
using System.Windows.Forms;
using NeutralNET.Framework.Connected;
using NeutralNET.Framework.Convolutional;
using NeutralNET.Framework.Neural.CNN;
using NeutralNET.Matrices;
using NeutralNET.Stuff;

namespace NeutralNET.ImageEpochViewer;

public partial class LetterWindow : Form
{
    private FlowLayoutPanel flowPanel;
    private System.Windows.Forms.Timer _timer;

    private static readonly string[] FontFamilies =
    [
        "Arial", "Times New Roman", "Georgia", "Verdana", "Tahoma",
        "Consolas", "Courier New", "Comic Sans MS", "Impact", "Trebuchet MS",
        "Palatino Linotype", "Segoe UI", "Lucida Console", "Garamond", "Century Gothic"
    ];

    private static readonly FontStyle[] SupportedStyles =
    [
        FontStyle.Regular,
        FontStyle.Bold,
        FontStyle.Italic,
        FontStyle.Bold | FontStyle.Italic
    ];

    private readonly CnnNetwork<Architecture> _network;
    private readonly List<(PictureBox Pic, Label Lbl, char TargetChar)> _letterSlots = [];

    public LetterWindow(CnnNetwork<Architecture> network)
    {
        _network = network;
        InitializeComponent();
        InitializeCustomLayout();

        RefreshAllLetters();
        StartTimer();
    }

    private void InitializeCustomLayout()
    {
        Width = 1040;
        Height = 720;
        Text = "CNN Letter Recognition (All A-Z Grid)";
        StartPosition = FormStartPosition.CenterScreen;
        BackColor = Color.FromArgb(18, 18, 18);

        flowPanel = new FlowLayoutPanel
        {
            Dock = DockStyle.Fill,
            AutoScroll = true,
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = true,
            Padding = new Padding(12),
            BackColor = Color.FromArgb(18, 18, 18)
        };

        Controls.Add(flowPanel);

        foreach (char targetChar in GraphicsUtils.DefaultLetters)
        {
            var itemPanel = new Panel
            {
                Width = 145,
                Height = 175,
                Margin = new Padding(6),
                BorderStyle = BorderStyle.FixedSingle,
                BackColor = Color.FromArgb(28, 28, 30)
            };

            var pic = new PictureBox
            {
                Width = 96,
                Height = 96,
                SizeMode = PictureBoxSizeMode.Zoom,
                Location = new Point((itemPanel.Width - 96) / 2, 8),
                BackColor = Color.Black,
                BorderStyle = BorderStyle.FixedSingle
            };

            var lbl = new Label
            {
                Width = 135,
                Height = 55,
                TextAlign = ContentAlignment.MiddleCenter,
                Location = new Point(5, 110),
                Font = new Font("Segoe UI", 9F, FontStyle.Bold),
                ForeColor = Color.White
            };

            itemPanel.Controls.Add(pic);
            itemPanel.Controls.Add(lbl);
            flowPanel.Controls.Add(itemPanel);

            _letterSlots.Add((pic, lbl, targetChar));
        }
    }

    private void StartTimer()
    {
        _timer = new System.Windows.Forms.Timer
        {
            Interval = 5000
        };
        _timer.Tick += (s, e) => RefreshAllLetters();
        _timer.Start();
    }

    protected override void OnFormClosed(FormClosedEventArgs e)
    {
        _timer?.Stop();
        _timer?.Dispose();
        base.OnFormClosed(e);
    }

    private unsafe void RefreshAllLetters()
    {
        foreach (var slot in _letterSlots)
        {
            char targetChar = slot.TargetChar;
            string randomFontName = FontFamilies[Random.Shared.Next(FontFamilies.Length)];
            FontStyle randomStyle = SupportedStyles[Random.Shared.Next(SupportedStyles.Length)];

            // 1. Generate the exact training-style pixel structure using GraphicsUtils
            using var font = new Font(randomFontName, GraphicsUtils.FontSize * GraphicsUtils.UpScale, randomStyle);
            var pixelStruct = GraphicsUtils.GenerateCharPixelStructRGB(targetChar, font, null);

            // 2. Create UI Bitmap and CNN input tensor
            Bitmap displayBmp = new Bitmap(GraphicsUtils.Width, GraphicsUtils.Height, PixelFormat.Format32bppArgb);

            int channels = 3;
            var inputMatrix = CnnMatrix.GetOrCreate(1, channels, GraphicsUtils.Height, GraphicsUtils.Width);
            float* pInput = inputMatrix.Pointer;

            int index = 0;
            for (int y = 0; y < GraphicsUtils.Height; y++)
            {
                for (int x = 0; x < GraphicsUtils.Width; x++)
                {
                    var rgb = pixelStruct.Values[index];
                    float r = rgb.R;
                    float g = rgb.G;
                    float b = rgb.B;

                    // Set pixel for UI display bitmap (32x32 native size)
                    displayBmp.SetPixel(x, y, Color.FromArgb((int)r, (int)g, (int)b));

                    // Populate CNN input tensor matching DataLoader's raw scale (do NOT divide by 255 if loader doesn't)
                    int spatialOffset = y * GraphicsUtils.Width + x;
                    pInput[0 * GraphicsUtils.PixelCount + spatialOffset] = r;
                    pInput[1 * GraphicsUtils.PixelCount + spatialOffset] = g;
                    pInput[2 * GraphicsUtils.PixelCount + spatialOffset] = b;

                    index++;
                }
            }

            slot.Pic.Image?.Dispose();
            slot.Pic.Image = displayBmp;

            // 3. Perform CNN Inference
            using NeuralMatrix output = _network.Forward(inputMatrix);
            inputMatrix.Dispose();

            int predictedClassIndex = 0;
            float maxConfidence = float.MinValue;

            float* pOutput = output.Pointer;
            int outputCols = output.UsedColumns;

            for (int i = 0; i < outputCols; i++)
            {
                float val = pOutput[i];
                if (val > maxConfidence)
                {
                    maxConfidence = val;
                    predictedClassIndex = i;
                }
            }

            char predictedChar = (predictedClassIndex >= 0 && predictedClassIndex < GraphicsUtils.DefaultLetters.Length)
                ? GraphicsUtils.DefaultLetters[predictedClassIndex]
                : '?';

            // 4. Color-coded evaluation
            if (predictedChar == targetChar && maxConfidence >= 0.7f)
            {
                slot.Lbl.Text = $"[{targetChar}] Pred: {predictedChar}\n({maxConfidence * 100:F1}%)";
                slot.Lbl.ForeColor = Color.LightGreen;
            }
            else if (predictedChar != targetChar)
            {
                slot.Lbl.Text = $"[{targetChar}] Pred: {predictedChar}\n({maxConfidence * 100:F1}%)";
                slot.Lbl.ForeColor = Color.IndianRed;
            }
            else
            {
                slot.Lbl.Text = $"[{targetChar}] Pred: {predictedChar}\n({maxConfidence * 100:F1}%)";
                slot.Lbl.ForeColor = Color.Gold;
            }
        }
    }
}
