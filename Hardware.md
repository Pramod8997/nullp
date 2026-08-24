  
    ESP32 GPIO 16 (RX2)  ──>  PZEM-004T TX
    ESP32 GPIO 17 (TX2)  ──>  PZEM-004T RX
    ESP32 GPIO 18        ──>  30A Relay Module IN (Active-LOW)
    ESP32 VIN (5V)       ──>  HLK-PM01 5V & Relay VCC & PZEM 5V
    ESP32 GND            ──>  Shared Common Ground
    Mains Live (L)       ──>  PZEM Terminal L & Relay COM -> Load
    Mains Neutral (N)    ──>  PZEM Terminal N & Load Return
    CT Clamp (100A)      ──>  Clamped around LIVE wire only

