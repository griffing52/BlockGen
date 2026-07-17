import keyboard
import time 
import webbrowser


a = 1824

while a < 19000:

    a = a+1
    siteExtension = "/download/action/?type=schematic"
    url = ("www.minecraft-schematics.com/schematic/" + str(a) + siteExtension)
    webbrowser.open(url)

    time.sleep(2)
    keyboard.press_and_release('ctrl+w')
    
    

