import torch
import torchvision

print('='*60)
print('PyTorch Environment Check')
print('='*60)
print(f'torch version: {torch.__version__}')
print(f'torchvision version: {torchvision.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
print('='*60)

# Check if versions match requirements
required_torch = '2.5.1'
required_torchvision = '0.20.1'

if torch.__version__.startswith(required_torch):
    print('✅ torch version is CORRECT')
else:
    print(f'❌ torch version is WRONG! Expected {required_torch}, got {torch.__version__}')

if torchvision.__version__.startswith(required_torchvision):
    print('✅ torchvision version is CORRECT')
else:
    print(f'❌ torchvision version is WRONG! Expected {required_torchvision}, got {torchvision.__version__}')

