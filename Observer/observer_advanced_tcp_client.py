#!/usr/bin/env python3
"""
Cliente TCP para AnyShake Observer - Visualización en Gals (cm/s²) a 250 Hz
Con filtro pasa bajas de 50 Hz, Energía Vectorial (RVM), detección STA/LTA y sistema de alertas
Versión con alta resolución y mejoras visuales
"""

import socket
import threading
import time
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
from scipy.signal import butter, lfilter
import math

# ===== CONFIGURACIÓN DE LA ESTACIÓN =====
STATION_ID = "0001"  # Identificador de la estación sísmica

# Constantes de conversión para el acelerómetro LSM6DSO32 (±4 g)
SENSITIVITY = 0.122
MG_TO_GALS = 0.980665
CONVERSION_FACTOR = SENSITIVITY * MG_TO_GALS
SAMPLING_RATE = 250

# Parámetros STA/LTA
STA_WINDOW = int(0.5 * SAMPLING_RATE)  # 0.5 segundos
LTA_WINDOW = int(10 * SAMPLING_RATE)   # 10 segundos

# Umbrales de detección
THR_ON = 3.0    # Umbral para iniciar anomalía
THR_OFF = 0.89  # Umbral para finalizar anomalía

# Tiempo mínimo de detección para evitar falsos positivos
MIN_ANOMALY_DURATION = 1.0  # Segundos

# Intervalo para reporte de máximos RVM durante anomalías
RVM_PEAK_REPORT_INTERVAL = 0.1  # 0.1 segundos

# Factor de división para RVM_PEAK
RVM_PEAK_DIVISION_FACTOR = 3.0

# ===== CONFIGURACIÓN DE VISUALIZACIÓN =====
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

# Colores personalizados
COLOR_TEAL = '#009688'  # Azul verdoso para componentes
COLOR_MAGENTA = '#E91E63'  # Magenta para RVM
COLOR_BLACK = '#000000'  # Negro para STA/LTA

class HybridFilter:
    def __init__(self, fs=SAMPLING_RATE, window_seconds=1, hp_cutoff=0.09, lp_cutoff=50):
        self.fs = fs
        self.window_size = window_seconds * fs
        
        self.moving_window = deque(maxlen=self.window_size)
        
        nyquist = 0.5 * fs
        normal_cutoff_hp = hp_cutoff / nyquist
        self.b_hp, self.a_hp = butter(2, normal_cutoff_hp, btype='high', analog=False)
        
        normal_cutoff_lp = lp_cutoff / nyquist
        self.b_lp, self.a_lp = butter(4, normal_cutoff_lp, btype='low', analog=False)
        
        self.zi_hp = np.zeros(max(len(self.a_hp), len(self.b_hp)) - 1)
        self.zi_lp = np.zeros(max(len(self.a_lp), len(self.b_lp)) - 1)
    
    def apply_filter(self, data):
        if not data:
            return data
        
        self.moving_window.extend(data)
        
        if self.moving_window:
            window_array = np.array(list(self.moving_window))
            p5 = np.percentile(window_array, 5)
            p95 = np.percentile(window_array, 95)
            midpoint = (p5 + p95) / 2
        else:
            midpoint = 0
            
        detrended = [x - midpoint for x in data]
        detrended_np = np.array(detrended, dtype=np.float64)
        
        filtered_hp, self.zi_hp = lfilter(self.b_hp, self.a_hp, detrended_np, zi=self.zi_hp)
        filtered_lp, self.zi_lp = lfilter(self.b_lp, self.a_lp, filtered_hp, zi=self.zi_lp)
        
        return filtered_lp.tolist()

class RecursiveSTA_LTA:
    """Implementación optimizada de STA/LTA recursivo para tiempo real"""
    def __init__(self, nsta=STA_WINDOW, nlta=LTA_WINDOW):
        self.nsta = nsta
        self.nlta = nlta
        self.csta = 1.0 / nsta
        self.clta = 1.0 / nlta
        self.icsta = 1.0 - self.csta
        self.iclta = 1.0 - self.clta
        self.sta = 0.0
        self.lta = 1.0  # Evitar división por cero
        self.sample_count = 0
        self.characteristic_function = deque(maxlen=5000)
        
    def update(self, value):
        self.sample_count += 1
        
        self.sta = self.csta * value + self.icsta * self.sta
        self.lta = self.clta * value + self.iclta * self.lta
        
        if self.lta < 1e-9:
            ratio = 0.0
        else:
            ratio = self.sta / self.lta
        
        self.characteristic_function.append(ratio)
        return ratio
    
    def get_cf_values(self, n_samples=None):
        if n_samples is None:
            return list(self.characteristic_function)
        return list(self.characteristic_function)[-n_samples:]

class SeismicAnomalyDetector:
    """Detección de anomalías sísmicas con reporte de picos RVM y PGA Horizontal"""
    def __init__(self, thr_on=THR_ON, thr_off=THR_OFF, min_duration=MIN_ANOMALY_DURATION):
        self.thr_on = thr_on
        self.thr_off = thr_off
        self.min_duration = min_duration
        
        self.anomaly_active = False
        self.anomaly_start_time = None
        self.last_ratio = 0.0
        self.anomaly_count = 0
        self.max_ratio_during_anomaly = 0.0
        
        # Para reporte de picos RVM
        self.last_rvm_peak_report_time = 0
        self.current_anomaly_max_rvm = 0.0
        self.rvm_peak_count = 0
        
        # Para cálculo de PGA Horizontal
        self.anomaly_enn_values = deque()
        self.anomaly_ene_values = deque()
        
        # Protección contra oscilaciones
        self.last_anomaly_end_time = None
        self.min_time_between_anomalies = 5.0
        
        # Control de frecuencia de mensajes
        self.last_message_time = 0
        
    def update_rvm_peak(self, rvm_value, timestamp):
        """
        Actualiza el máximo de RVM durante una anomalía activa
        y reporta cada RVM_PEAK_REPORT_INTERVAL segundos
        """
        if not self.anomaly_active:
            return
            
        # Actualizar máximo de RVM durante esta anomalía
        if rvm_value > self.current_anomaly_max_rvm:
            self.current_anomaly_max_rvm = rvm_value
        
        # Verificar si es tiempo de reportar un pico
        current_time = time.time()
        if current_time - self.last_rvm_peak_report_time >= RVM_PEAK_REPORT_INTERVAL:
            if self.current_anomaly_max_rvm > 0:
                # Dividir RVM por el factor especificado antes de reportar
                rvm_divided = self.current_anomaly_max_rvm / RVM_PEAK_DIVISION_FACTOR
                
                # Reportar el pico máximo (redondeado a 2 decimales)
                utc_time = timestamp.astimezone(timezone.utc)
                time_str = utc_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                print(f"ID_SEISMIC_STATION = {STATION_ID}: RVM_PEAK: {round(rvm_divided, 2)} Gals at {time_str}")
                
                self.rvm_peak_count += 1
                self.last_rvm_peak_report_time = current_time
                # Resetear para el próximo intervalo
                self.current_anomaly_max_rvm = 0.0
    
    def update_horizontal_components(self, enn_value, ene_value):
        """
        Almacena valores de las componentes horizontales durante una anomalía
        para cálculo posterior de PGA Horizontal
        """
        if not self.anomaly_active:
            return
            
        self.anomaly_enn_values.append(enn_value)
        self.anomaly_ene_values.append(ene_value)
    
    def calculate_pga_horizontal(self):
        """
        Calcula PGA Horizontal: sqrt(max(ENN²) + max(ENE²)) / 2
        """
        if len(self.anomaly_enn_values) == 0 or len(self.anomaly_ene_values) == 0:
            return 0.0
        
        # Calcular máximos de las componentes horizontales
        max_enn = max([abs(x) for x in self.anomaly_enn_values])
        max_ene = max([abs(x) for x in self.anomaly_ene_values])
        
        # Aplicar fórmula PGA_h = sqrt(max(ENN²) + max(ENE²)) / 2
        pga_h = math.sqrt(max_enn**2 + max_ene**2) / 2.0
        
        return round(pga_h, 2)  # Redondear a 2 decimales
    
    def update(self, ratio, timestamp):
        """
        Actualiza el detector con nuevo ratio STA/LTA
        Retorna: (event_detected, message)
        """
        self.last_ratio = ratio
        
        # Control de frecuencia de mensajes - máximo 1 mensaje por segundo
        current_time = time.time()
        if current_time - self.last_message_time < 1.0:
            return False, None
        
        # Si hay una anomalía activa, actualizar el ratio máximo
        if self.anomaly_active:
            if ratio > self.max_ratio_during_anomaly:
                self.max_ratio_during_anomaly = ratio
        
        # Verificar si debemos iniciar una nueva anomalía
        if not self.anomaly_active and ratio >= self.thr_on:
            # Verificar que haya pasado suficiente tiempo desde la última anomalía
            if self.last_anomaly_end_time is not None:
                time_since_last = (timestamp - self.last_anomaly_end_time).total_seconds()
                if time_since_last < self.min_time_between_anomalies:
                    return False, None
            
            # Iniciar anomalía
            self.anomaly_active = True
            self.anomaly_start_time = timestamp
            self.max_ratio_during_anomaly = ratio
            
            # Inicializar contadores para reporte de picos RVM
            self.last_rvm_peak_report_time = time.time()
            self.current_anomaly_max_rvm = 0.0
            self.rvm_peak_count = 0
            
            # Inicializar buffers para componentes horizontales
            self.anomaly_enn_values.clear()
            self.anomaly_ene_values.clear()
            
            utc_time = timestamp.astimezone(timezone.utc)
            time_str = utc_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            message = f"Anomaly_Start in {time_str}"
            
            self.last_message_time = current_time
            return True, message
        
        # Verificar si debemos finalizar la anomalía actual
        elif self.anomaly_active and ratio < self.thr_off:
            # Verificar que la anomalía haya durado al menos el tiempo mínimo
            anomaly_duration = (timestamp - self.anomaly_start_time).total_seconds()
            if anomaly_duration >= self.min_duration:
                # Reportar último pico RVM si queda alguno
                if self.current_anomaly_max_rvm > 0:
                    utc_time = timestamp.astimezone(timezone.utc)
                    time_str = utc_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    rvm_divided = self.current_anomaly_max_rvm / RVM_PEAK_DIVISION_FACTOR
                    print(f"ID_SEISMIC_STATION = {STATION_ID}: RVM_PEAK: {round(rvm_divided, 2)} Gals at {time_str}")
                
                # Calcular PGA Horizontal
                pga_h = self.calculate_pga_horizontal()
                
                # Finalizar anomalía
                self.anomaly_active = False
                self.last_anomaly_end_time = timestamp
                self.anomaly_count += 1
                
                utc_time = timestamp.astimezone(timezone.utc)
                time_str = utc_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                message = f"Anomaly_End in {time_str} (Duration: {anomaly_duration:.2f}s, Max Ratio: {round(self.max_ratio_during_anomaly, 2)}, RVM Peaks: {self.rvm_peak_count}, PGA_h: {pga_h} Gals)"
                
                self.anomaly_start_time = None
                self.max_ratio_during_anomaly = 0.0
                self.current_anomaly_max_rvm = 0.0
                self.last_message_time = current_time
                return True, message
            else:
                # Anomalía demasiado corta, ignorarla
                self.anomaly_active = False
                self.anomaly_start_time = None
                self.max_ratio_during_anomaly = 0.0
                self.current_anomaly_max_rvm = 0.0
                self.anomaly_enn_values.clear()
                self.anomaly_ene_values.clear()
                return False, None
        
        return False, None
    
    def get_status(self):
        """Retorna el estado actual del detector"""
        return {
            'anomaly_active': self.anomaly_active,
            'current_ratio': self.last_ratio,
            'anomaly_count': self.anomaly_count,
            'start_time': self.anomaly_start_time,
            'current_max_rvm': self.current_anomaly_max_rvm,
            'rvm_peak_count': self.rvm_peak_count,
            'horizontal_samples': len(self.anomaly_enn_values)
        }

class AnyShakeRealTimePlotter:
    def __init__(self, host='localhost', port=30000, display_seconds=30):
        self.host = host
        self.port = port
        self.display_seconds = display_seconds
        self.running = False
        self.socket = None
        self.data_buffer = b''
        
        # STA/LTA para detección de eventos
        self.sta_lta = RecursiveSTA_LTA()
        self.anomaly_detector = SeismicAnomalyDetector()
        
        # Filtros para cada componente
        self.filters = {
            'ENZ': HybridFilter(window_seconds=1, hp_cutoff=0.09, lp_cutoff=50),
            'ENE': HybridFilter(window_seconds=1, hp_cutoff=0.09, lp_cutoff=50),
            'ENN': HybridFilter(window_seconds=1, hp_cutoff=0.09, lp_cutoff=50)
        }
        
        # Configuración de la visualización - 5 subgráficas con alta resolución
        self.fig, self.axes = plt.subplots(5, 1, figsize=(16, 14), dpi=120)
        self.ax1, self.ax2, self.ax3, self.ax4, self.ax5 = self.axes
        
        # Título principal actualizado
        self.fig.suptitle(f'SKYALERT SEISMIC STATION {STATION_ID} - AnyShake - {SAMPLING_RATE} Hz - RVM + STA/LTA + Anomaly Detection', 
                         fontsize=16, fontweight='bold')
        
        # Buffers de datos
        max_points = display_seconds * SAMPLING_RATE
        self.times = deque(maxlen=max_points)
        self.data_enz = deque(maxlen=max_points)
        self.data_ene = deque(maxlen=max_points)
        self.data_enn = deque(maxlen=max_points)
        
        # Buffer para STA/LTA
        self.sta_lta_values = deque(maxlen=max_points)
        
        # Líneas de gráfico con nuevos colores
        self.line_enz, = self.ax1.plot([], [], color=COLOR_TEAL, linewidth=1.0)
        self.line_ene, = self.ax2.plot([], [], color=COLOR_TEAL, linewidth=1.0)
        self.line_enn, = self.ax3.plot([], [], color=COLOR_TEAL, linewidth=1.0)
        self.line_rvm, = self.ax4.plot([], [], color=COLOR_MAGENTA, linewidth=1.2)
        self.line_sta_lta, = self.ax5.plot([], [], color=COLOR_BLACK, linewidth=1.2)
        
        # Configurar ejes con mejoras de tamaño de letra
        components_config = [
            (self.ax1, 'Componente Vertical (ENZ)', COLOR_TEAL),
            (self.ax2, 'Componente Este-Oeste (ENE)', COLOR_TEAL),
            (self.ax3, 'Componente Norte-Sur (ENN)', COLOR_TEAL),
            (self.ax4, 'Energía Vectorial (RVM)', COLOR_MAGENTA),
            (self.ax5, f'STA/LTA Ratio (Thr_ON={THR_ON}, Thr_OFF={THR_OFF})', COLOR_BLACK)
        ]
        
        for ax, title, color in components_config:
            if ax == self.ax5:
                ax.set_ylabel('Ratio STA/LTA', fontsize=12)
            else:
                ax.set_ylabel('Aceleración (Gals)', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # Mejorar tamaño de ticks
            ax.tick_params(axis='both', which='major', labelsize=10)
        
        self.ax5.set_xlabel('Tiempo (segundos)', fontsize=12)
        
        # Umbrales de detección con mejor visibilidad
        self.ax5.axhline(y=THR_ON, color='r', linestyle='--', alpha=0.8, linewidth=2, label=f'Thr_ON={THR_ON}')
        self.ax5.axhline(y=THR_OFF, color='g', linestyle='--', alpha=0.8, linewidth=2, label=f'Thr_OFF={THR_OFF}')
        self.ax5.legend(fontsize=10, loc='upper right')
        
        # Texto para mostrar estado de anomalía con mejor visibilidad
        self.anomaly_text = self.ax5.text(0.02, 0.95, '', transform=self.ax5.transAxes, 
                                         fontsize=11, fontweight='bold', verticalalignment='top',
                                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        self.lock = threading.Lock()
        
        # Historial de máximos para ajuste suave de límites
        self.max_values_history = {
            'ENZ': deque(maxlen=10),
            'ENE': deque(maxlen=10),
            'ENN': deque(maxlen=10),
            'RVM': deque(maxlen=10),
            'STA/LTA': deque(maxlen=10)
        }
        
    def calculate_dynamic_ylimits(self, data, component_name, is_positive_only=False):
        """
        Calcula límites dinámicos para el eje Y basado en los datos actuales
        """
        if len(data) == 0:
            if is_positive_only:
                return 0, 1
            else:
                return -1, 1
        
        # Calcular el máximo absoluto de los datos actuales
        current_max = np.max(np.abs(data))
        
        # Agregar a historial
        self.max_values_history[component_name].append(current_max)
        
        # Calcular percentil 90 del historial para suavizar cambios bruscos
        if len(self.max_values_history[component_name]) > 0:
            historical_max = np.percentile(list(self.max_values_history[component_name]), 90)
            # Usar el máximo entre el actual y el histórico
            target_max = max(current_max, historical_max)
        else:
            target_max = current_max
        
        # Añadir margen del 25%
        margin = 1.25
        limit = target_max * margin
        
        # Establecer límites mínimos
        min_limit = 0.1  # Mínimo de 0.1 Gal para visualización
        
        if limit < min_limit:
            limit = min_limit
        
        if is_positive_only:
            # Para RVM (siempre positivo)
            return 0, limit
        else:
            # Para componentes de aceleración (simétrico alrededor de 0)
            return -limit, limit
    
    def convert_to_gals(self, data):
        return [x * CONVERSION_FACTOR for x in data]
    
    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.host, self.port))
            print(f"Conectado a {self.host}:{self.port}")
            print(f"ID de estación: {STATION_ID}")
            print(f"Sampling rate: {SAMPLING_RATE} Hz")
            print(f"STA window: {STA_WINDOW} muestras ({STA_WINDOW/SAMPLING_RATE:.1f} s)")
            print(f"LTA window: {LTA_WINDOW} muestras ({LTA_WINDOW/SAMPLING_RATE:.1f} s)")
            print(f"Umbrales - Thr_ON: {THR_ON}, Thr_OFF: {THR_OFF}")
            print(f"Duración mínima de anomalía: {MIN_ANOMALY_DURATION}s")
            print(f"Reporte de picos RVM cada: {RVM_PEAK_REPORT_INTERVAL}s durante anomalías")
            print(f"Factor de división RVM: {RVM_PEAK_DIVISION_FACTOR}")
            print("Sistema de detección de anomalías activado")
            return True
        except Exception as e:
            print(f"Error de conexión: {e}")
            return False
    
    def parse_packet(self, packet):
        try:
            parts = packet.strip().split(',')
            
            if len(parts) < 9 or parts[0][0] != '$' or '*' not in parts[-1]:
                return None
            
            component = parts[4]
            timestamp = int(parts[5])
            sampling_rate = int(parts[6])
            
            if sampling_rate != SAMPLING_RATE:
                return None
            
            data_start = 7
            data_end = len(parts) - 1
            raw_data = parts[data_start:data_end]
            data_values = [int(x) for x in raw_data]
            
            if data_values and component in self.filters:
                data_values = self.filters[component].apply_filter(data_values)
                data_values = self.convert_to_gals(data_values)
            
            return {
                'component': component,
                'timestamp': timestamp,
                'sampling_rate': sampling_rate,
                'data': data_values
            }
            
        except Exception as e:
            return None
    
    def process_packet(self, packet_data):
        if not packet_data:
            return
        
        component = packet_data['component']
        data_values = packet_data['data']
        current_time = time.time()
        
        with self.lock:
            for i, value in enumerate(data_values):
                sample_time = current_time - (len(data_values) - i) * (1.0 / SAMPLING_RATE)
                
                if component == 'ENZ':
                    self.times.append(sample_time)
                    self.data_enz.append(value)
                elif component == 'ENE':
                    self.data_ene.append(value)
                elif component == 'ENN':
                    self.data_enn.append(value)
    
    def receive_data(self):
        self.running = True
        self.data_buffer = b''
        
        try:
            while self.running:
                try:
                    data = self.socket.recv(1024)
                    if not data:
                        break
                    
                    self.data_buffer += data
                    
                    while b'\r' in self.data_buffer:
                        packet, self.data_buffer = self.data_buffer.split(b'\r', 1)
                        
                        try:
                            packet_str = packet.decode('ascii', errors='ignore').strip()
                            if packet_str:
                                parsed_data = self.parse_packet(packet_str)
                                if parsed_data:
                                    self.process_packet(parsed_data)
                        except Exception:
                            continue
                            
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"Error en recepción: {e}")
                    break
                        
        except Exception as e:
            print(f"Error crítico: {e}")
        finally:
            self.running = False
    
    def calculate_rvm(self, enz_array, ene_array, enn_array):
        return np.sqrt(enz_array**2 + ene_array**2 + enn_array**2)
    
    def update_plot(self, frame):
        with self.lock:
            times_copy = np.array(self.times)
            enz_copy = np.array(self.data_enz)
            ene_copy = np.array(self.data_ene)
            enn_copy = np.array(self.data_enn)
        
        min_len = min(len(times_copy), len(enz_copy), len(ene_copy), len(enn_copy))
        if min_len < 10:
            return self.line_enz, self.line_ene, self.line_enn, self.line_rvm, self.line_sta_lta
        
        # Recortar a longitud común
        times_trim = times_copy[-min_len:]
        enz_trim = enz_copy[-min_len:]
        ene_trim = ene_copy[-min_len:]
        enn_trim = enn_copy[-min_len:]
        
        # Calcular RVM para todas las muestras
        rvm_data = self.calculate_rvm(enz_trim, ene_trim, enn_trim)
        
        # Calcular STA/LTA para cada valor de RVM
        sta_lta_data = []
        current_time = datetime.now()
        
        for i, rvm_value in enumerate(rvm_data):
            ratio = self.sta_lta.update(rvm_value)
            sta_lta_data.append(ratio)
            
            # Actualizar detector de picos RVM durante anomalías
            self.anomaly_detector.update_rvm_peak(rvm_value, current_time)
            
            # Actualizar componentes horizontales para cálculo de PGA_h
            if i < len(ene_trim) and i < len(enn_trim):
                self.anomaly_detector.update_horizontal_components(enn_trim[i], ene_trim[i])
        
        # Actualizar el buffer de STA/LTA para visualización
        self.sta_lta_values.extend(sta_lta_data)
        
        # Verificar detección de anomalías solo con el último valor
        if len(sta_lta_data) > 0:
            latest_ratio = sta_lta_data[-1]
            event_detected, message = self.anomaly_detector.update(latest_ratio, current_time)
            if event_detected and message:
                print(f"ID_SEISMIC_STATION = {STATION_ID}: {message}")
        
        # Tiempo relativo para visualización
        current_time_sec = time.time()
        rel_times = current_time_sec - times_trim
        
        # Asegurar que STA/LTA tenga la misma longitud que los tiempos
        sta_lta_trim = list(self.sta_lta_values)[-min_len:]
        if len(sta_lta_trim) < min_len:
            sta_lta_trim = [0] * (min_len - len(sta_lta_trim)) + sta_lta_trim
        
        # Actualizar gráficos
        self.line_enz.set_data(rel_times, enz_trim)
        self.line_ene.set_data(rel_times, ene_trim)
        self.line_enn.set_data(rel_times, enn_trim)
        self.line_rvm.set_data(rel_times, rvm_data)
        self.line_sta_lta.set_data(rel_times, sta_lta_trim)
        
        # Ajustar límites X
        xlim = (self.display_seconds, 0)
        for ax in self.axes:
            ax.set_xlim(xlim)
        
        # Ajustar límites Y dinámicamente
        ylim_enz = self.calculate_dynamic_ylimits(enz_trim, 'ENZ', False)
        ylim_ene = self.calculate_dynamic_ylimits(ene_trim, 'ENE', False)
        ylim_enn = self.calculate_dynamic_ylimits(enn_trim, 'ENN', False)
        ylim_rvm = self.calculate_dynamic_ylimits(rvm_data, 'RVM', True)
        ylim_sta_lta = self.calculate_dynamic_ylimits(sta_lta_trim, 'STA/LTA', True)
        
        self.ax1.set_ylim(ylim_enz)
        self.ax2.set_ylim(ylim_ene)
        self.ax3.set_ylim(ylim_enn)
        self.ax4.set_ylim(ylim_rvm)
        self.ax5.set_ylim(ylim_sta_lta)
        
        # Actualizar texto de estado de anomalía
        status = self.anomaly_detector.get_status()
        if status['anomaly_active']:
            anomaly_text = f"ANOMALÍA ACTIVA\nRatio: {status['current_ratio']:.2f}\nRVM Peaks: {status['rvm_peak_count']}\nMuestras: {status['horizontal_samples']}"
            color = 'red'
        else:
            anomaly_text = f"Estado normal\nRatio: {status['current_ratio']:.2f}\nEventos: {status['anomaly_count']}"
            color = 'green'
        
        self.anomaly_text.set_text(anomaly_text)
        self.anomaly_text.set_color(color)
        
        # Cambiar color de fondo según estado
        if status['anomaly_active']:
            self.ax5.set_facecolor('#FFF0F0')
        else:
            self.ax5.set_facecolor('#F0FFF0')
        
        return self.line_enz, self.line_ene, self.line_enn, self.line_rvm, self.line_sta_lta, self.anomaly_text
    
    def start(self):
        if not self.connect():
            return False
        
        self.receiver_thread = threading.Thread(target=self.receive_data)
        self.receiver_thread.daemon = True
        self.receiver_thread.start()
        
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, interval=150, blit=True, cache_frame_data=False
        )
        
        print(f"SKYALERT SEISMIC STATION {STATION_ID} - Sistema de monitoreo iniciado")
        print("Esperando detección de anomalías...")
        plt.show()
        return True
    
    def stop(self):
        self.running = False
        if self.socket:
            self.socket.close()
        
        # Reporte final
        status = self.anomaly_detector.get_status()
        print(f"\n--- REPORTE FINAL - SKYALERT SEISMIC STATION {STATION_ID} ---")
        print(f"Total de eventos detectados: {status['anomaly_count']}")
        if status['anomaly_active']:
            print("¡ATENCIÓN: Anomalía activa al detener el sistema!")
        
        print("Sistema detenido")

def main():
    plotter = AnyShakeRealTimePlotter(display_seconds=20)
    
    try:
        if plotter.start():
            while plt.fignum_exists(plotter.fig.number):
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nInterrupción por teclado recibida")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        plotter.stop()

if __name__ == "__main__":
    main()
