import os
import sys
from config import Config

print("--- Verifying Security Fixes ---")

# 1. Check Secrets
print("\nChecking Secrets configuration...")
# Simulate production environment
os.environ['FLASK_ENV'] = 'production'
os.environ['RENDER'] = 'true'

# Reset Config class to reload from env
import importlib
import config
importlib.reload(config)
from config import Config

if Config.SECRET_KEY is None:
    print("✅ SECRET_KEY is None in production (correctly removed default)")
else:
    print(f"❌ SECRET_KEY has a value: {Config.SECRET_KEY}")

if Config.ADMIN_PASSWORD is None:
    print("✅ ADMIN_PASSWORD is None in production (correctly removed default)")
else:
    print(f"❌ ADMIN_PASSWORD has a value: {Config.ADMIN_PASSWORD}")

# 2. Check Security Warning
print("\nChecking Security Warning function...")
try:
    Config.check_production_security()
    print("✅ check_production_security ran (check output for warnings)")
except Exception as e:
    print(f"❌ check_production_security failed: {e}")

print("\n--- Verification Complete ---")
