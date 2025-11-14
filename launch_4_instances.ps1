# Script de PowerShell para ejecutar 4 instancias de entrenamiento visual en paralelo
# Cada una se ejecutará en su propia ventana organizadas en cuadrícula

Write-Host "Iniciando 4 instancias de entrenamiento en paralelo (30+ minutos)..." -ForegroundColor Green
Write-Host ""

# Obtener dimensiones de la pantalla
Add-Type -AssemblyName System.Windows.Forms
$screen = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea
$halfWidth = [math]::Floor($screen.Width / 2)
$halfHeight = [math]::Floor($screen.Height / 2)

# Posiciones para cuadrícula 2x2
$positions = @(
    @{X=0; Y=0; Width=$halfWidth; Height=$halfHeight},  # Superior izquierda
    @{X=$halfWidth; Y=0; Width=$halfWidth; Height=$halfHeight},  # Superior derecha
    @{X=0; Y=$halfHeight; Width=$halfWidth; Height=$halfHeight},  # Inferior izquierda
    @{X=$halfWidth; Y=$halfHeight; Width=$halfWidth; Height=$halfHeight}  # Inferior derecha
)

# Crear 4 ventanas de PowerShell, cada una ejecutando train_visual.py
# 500,000 timesteps = aproximadamente 30-40 minutos con 4 instancias visuales
for ($i=1; $i -le 4; $i++) {
    $title = "Pokemon RL - Instancia $i"
    $pos = $positions[$i-1]
    
    Write-Host "  Lanzando instancia $i..." -ForegroundColor Cyan
    
    $process = Start-Process powershell -ArgumentList "-NoExit", "-Command", @"
`$Host.UI.RawUI.WindowTitle = '$title'
Write-Host 'Instancia $i de Pokemon RL - Entrenando por 30+ minutos' -ForegroundColor Green
Write-Host ''
cd 'C:\Users\tomas.s.vicente\Documents\Apps\llmpoke'
python train_visual.py --timesteps 500000 --save-freq 50000
"@ -PassThru
    
    # Esperar a que se cree la ventana
    Start-Sleep -Milliseconds 1500
    
    # Posicionar la ventana de PowerShell
    try {
        $hwnd = $process.MainWindowHandle
        if ($hwnd -ne 0) {
            Add-Type @"
                using System;
                using System.Runtime.InteropServices;
                public class Win32 {
                    [DllImport("user32.dll")]
                    [return: MarshalAs(UnmanagedType.Bool)]
                    public static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, uint uFlags);
                }
"@
            $SWP_NOZORDER = 0x0004
            [Win32]::SetWindowPos($hwnd, [IntPtr]::Zero, $pos.X, $pos.Y, $pos.Width, $pos.Height, $SWP_NOZORDER)
        }
    } catch {
        Write-Host "  No se pudo posicionar automáticamente la ventana $i" -ForegroundColor Yellow
    }
    
    Start-Sleep -Milliseconds 500
}

Write-Host ""
Write-Host "4 instancias lanzadas!" -ForegroundColor Green
Write-Host ""
Write-Host "Veras 4 ventanas de PowerShell, cada una con su emulador" -ForegroundColor Yellow
Write-Host "Cada instancia entrena independientemente" -ForegroundColor Yellow
Write-Host "Cierra cada ventana individualmente cuando quieras detenerlas" -ForegroundColor Yellow
