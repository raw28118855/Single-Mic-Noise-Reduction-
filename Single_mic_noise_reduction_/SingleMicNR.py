import numpy as np
import numpy.matlib
import math
import cmath
class SingleMicNR:
    def __init__(self,pcm_data,sample_rate):
        self.pcm_data=pcm_data
        self.sample_rate=sample_rate
        self.IS =.25   # seconds
        self.W = int(.025 * self.sample_rate) # Window length is 25 ms
        self.nfft=self.W
        self.SP = .4  # Shift percentage is 40 % (10ms) % Overlap - Add method works good with this value(.4)
        self.wnd=np.hamming(self.W)

    def Magnitude_Subtraction(self):   #SSBoll79
        NIS = int((self.IS * self.sample_rate - self.W) / (self.SP * self.W) + 1) # number of initial silence segments
        Gamma = 1 # Magnitude Power(1 for magnitude spectral subtraction 2 for power spectrum subtraction)
        y = self.segment(self.pcm_data,self.W,self.SP,self.wnd)
        #Y = y + 0j
        Y=y+0j
        Yx, Yy = Y.shape
        Y=Y.T
        Y = np.fft.fft(Y,self.nfft)
        Y=Y.T
        YPhase = np.angle(Y[0:int(Yx / 2)+1,:].T)  # Noisy Speech Phase
        YPhase=YPhase.T
        Y = np.dot(abs(Y[0:int(Yx / 2)+1,:].T),Gamma) # Specrogram
        Y=Y.T
        FreqResol,numberOfFrames = Y.shape

        N=np.zeros([1,FreqResol])

        for i in range(FreqResol):
            a=Y[i,0:NIS]
            a = a.reshape(1, a.shape[0])
            N[:,i] = np.mean(a)   # initial Noise Power Spectrum mean
        N = N.reshape(N.shape[1], 1)
        NRM = np.zeros(np.shape(N))  # Noise Residual Maximum(Initialization)
        NoiseCounter = 0
        NoiseLength = 9 # This is a smoothing factor for the noise updating
        Beta=.1
        YS=np.zeros(Y.shape) # Y Magnitude Averaged

        buf=np.zeros(N.shape)
        X=np.zeros(Y.shape)
        for i in range(1,(numberOfFrames-1)):
            YS[:,i]=(Y[:, i-1] + Y[:, i]+Y[:, i + 1])/3
        YS[:,0]=Y[:,0]
        YS[:,-1]=Y[:,-1]
            #buf=buf.reshape(buf.shape[0],1)
        for i in range(0,numberOfFrames-1):
            NoiseFlag, SpeechFlag, NoiseCounter, Dist = self.vad(np.dot(Y[:, i], (1 / Gamma)), np.dot(N, ( 1 / Gamma)), NoiseCounter) # Magnitude

            if SpeechFlag == 0:
                buf=Y[:,i]
                buf = buf.reshape(buf.shape[0], 1)
                N = (NoiseLength * N +buf) / (NoiseLength + 1) # Update and smooth noise
                buf=YS[:,i]
                buf = buf.reshape(buf.shape[0], 1)
                NRM = np.maximum(NRM, buf-N) # Update Maximum Noise Residue
                X[:, i]=Beta * Y[:, i]
            else:
                buf = YS[:, i]
                buf = buf.reshape(buf.shape[0], 1)
                D = buf-N # Specral Subtraction
                if (i > 0 and i < numberOfFrames): # Residual Noise Reduction
                    for j in range (0,len(D)-1):
                        if D[j,0] < NRM[j]:
                            D[j,0] =min(D[j,0], YS[j, i - 1]-N[j], YS[j,i + 1]-N[j])
                for k in range(FreqResol):
                    X[k, i]=np.maximum(D[k,0],0)

        output = self.OverlapAdd2(np.dot(X,(1 / Gamma)), YPhase, self.W, self.SP * self.W)
        return output

    def vad(self,data,noise,NoiseCounter):
        NoiseMargin = 3.5
        Hangover = 8
        FreqResol = len(data)
        data = data.reshape(data.shape[0], 1)
        SpectralDist = 20 * (np.log10(data) - np.log10(noise))
        SpectralDist[SpectralDist < 0] = 0

        Dist = np.mean(SpectralDist)
        print(Dist,NoiseMargin,NoiseCounter)
        if (Dist < NoiseMargin):
            NoiseFlag = 1
            NoiseCounter = NoiseCounter + 1
        else:
            NoiseFlag = 0
            NoiseCounter = 0
        if (NoiseCounter > Hangover):
            SpeechFlag = 0
        else:
            SpeechFlag = 1
        return NoiseFlag, SpeechFlag, NoiseCounter, Dist
    def OverlapAdd2(self,XNEW,yphase,windowLen,ShiftLen):
        if int(ShiftLen) != ShiftLen:
            ShiftLen=int(ShiftLen)

        FreqRes,FrameNum = np.shape(XNEW)

        #Spec = np.dot(XNEW, math.exp(yphase * cmath.sqrt(-1)))
        virtual=0+1j
        #Spec = np.dot(XNEW, aaa)
        Spec = XNEW*np.exp(yphase * virtual)

        if  windowLen % 2: #if FreqResol is odd
            Spec=np.vstack((Spec,np.flipud(np.conj(Spec[1::,:]))))
        else:
            Spec = np.vstack((Spec, np.flipud(np.conj(Spec[1:-1, :]))))
        sig = np.zeros([int((FrameNum - 1) * ShiftLen + windowLen), 1])
        weight = sig
        spec = np.zeros([windowLen, 1])
        for i in range (0,FrameNum):
            start = i * ShiftLen
            spec = Spec[:, i]
            #spec = spec.reshape(spec.shape[0], 1)
            spec=np.fft.ifft(spec,n=windowLen)
            spec=spec.real
            spec = spec.reshape(spec.shape[0], 1)
            sig[int(start): int(start + windowLen)] =sig[int(start): int(start + windowLen)] + spec
        ReconstructedSignal = sig
        return ReconstructedSignal
    def segment(self,signal,W,SP,wnd):
        # SEGMENT chops a signal to overlapping windowed segments
        # A = SEGMENT(X, W, SP, WIN) returns a matrix which its columns are segmented
        # and windowed frames of the input one dimentional  signal, X.W is the
        # number of samples per window, default value W = 256. SP is the shift
        # percentage, default value SP = 0.4.WIN is the window that is multiplied by
        # each segment and its length should be W.the default window is hamming window.
        Window= np.mat(wnd)
        Window = Window.transpose() # make it a column vector
        L = len(signal)

        SP = int(W* SP)

        N = int((L - W) / SP + 1) # number of segments

        Index = np.array(np.matlib.repmat(range(1,W+1),N,1)+ np.matlib.repmat(np.mat(range(0, N)).transpose()*SP, 1, W)).transpose()
        Index=Index-1
        hw = np.array(np.matlib.repmat(Window, 1, N))
        Seg = np.multiply(np.array(signal)[Index],hw)
        return Seg