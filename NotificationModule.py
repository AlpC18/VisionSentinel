import os
import time
import subprocess
from twilio.rest import Client

class NotificationSystem:
    def __init__(self):
        # -----------------------------------------------------------
        # AYARLAR: BURAYA TELEFON NUMARANIZI YAZIN
        # -----------------------------------------------------------
        self.my_phone_number = "+905xxxxxxxxx"  # Ornegin: +905321234567
        # -----------------------------------------------------------
        
        # Twilio ayarlari (Opsiyonel - iMessage calismazsa diye)
        self.account_sid = 'ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
        self.auth_token = 'your_auth_token_here'
        self.twilio_number = '+10000000000'

        self.last_sent_time = 0
        self.cooldown_seconds = 60 # 1 dakika bekle

    def send_alert(self, message_body):
        current_time = time.time()
        if (current_time - self.last_sent_time) < self.cooldown_seconds:
            return False # Cooldown

        # 1. Yontem: Mac 'Mesajlar' uygulamasi uzerinden (UCRETSIZ & HIZLI)
        # Sadece Mac kullanicilari icin gecerlidir.
        if self._try_send_imessage(message_body):
            print(f"[BILDIRIM] iMessage/SMS gonderildi: {self.my_phone_number}")
            self.last_sent_time = current_time
            return True
            
        # 2. Yontem: Twilio (Yedek)
        if self._try_send_twilio(message_body):
            self.last_sent_time = current_time
            return True
            
        return False

    def _try_send_imessage(self, message):
        # Telefon numarasi girilmemisse deneme
        if "xxxx" in self.my_phone_number:
            print("[HATA] Lutfen NotificationModule.py dosyasini acip telefon numaranizi girin!")
            return False

        try:
            # AppleScript ile Mesajlar uygulamasini tetikle
            # Bu yontem Mac'inize tanimli iCloud hesabini kullanir.
            script = f'''
            tell application "Messages"
                set targetBuddy to "{self.my_phone_number}"
                set targetService to id of 1st service whose service type = iMessage
                set textMessage to "{message}"
                send textMessage to participant targetBuddy
            end tell
            '''
            
            # Subprocess ile calistir
            subprocess.run(["osascript", "-e", script], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception as e:
            print(f"[iMessage Hatasi] Gonderilemedi (Numara rehberde kayitli olmali veya iMessage aktif olmali): {e}")
            return False

    def _try_send_twilio(self, message):
        if 'XX' in self.account_sid: return False
        try:
            client = Client(self.account_sid, self.auth_token)
            client.messages.create(
                body=message,
                from_=self.twilio_number,
                to=self.my_phone_number
            )
            print("[Twilio] SMS Gonderildi.")
            return True
        except:
            return False
