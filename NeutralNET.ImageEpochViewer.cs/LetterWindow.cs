using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using NeutralNET.Framework.Connected;
using NeutralNET.Framework.Neural.CNN;
using NeutralNET.Matrices;
using NeutralNET.Stuff;
using NeutralNET.Test.Data; // Required for LetterDataLoader

namespace NeutralNET.ImageEpochViewer;

public partial class LetterWindow : Form
{
    private FlowLayoutPanel flowPanel;
    private System.Windows.Forms.Timer _timer;

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
                Width = GraphicsUtils.Width,
                Height = GraphicsUtils.Height,
                SizeMode = PictureBoxSizeMode.Zoom,
                Location = new Point((itemPanel.Width - GraphicsUtils.Width) / 2, 8),
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
            var (inputMatrix, displayBmp) = LetterDataLoader.GenerateSampleForUI(targetChar);

            slot.Pic.Image?.Dispose();
            slot.Pic.Image = displayBmp;

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

            // Color-coded evaluation
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
