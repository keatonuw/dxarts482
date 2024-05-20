import signalflow as sf

graph = sf.AudioGraph()

srcs = [sf.Buffer("./Samples/clippy.wav"), sf.Buffer("./Samples/OVERTONE.wav")]
kern = sf.Buffer("./Samples/SNAP.wav")

fb_buf = sf.Buffer(2, graph.sample_rate)
resample_buf = sf.Buffer(2, graph.sample_rate)

state = sf.RandomImpulse(0.1)
clock = sf.RandomImpulse(sf.Sequence([30.0, 30.0, 15.0, 60.0], clock=state))
granulator = [
    sf.Granulator(
        buffer=b,
        clock=clock,
        pos=sf.RandomBrownian(0.0, 1.0, state),
        duration=sf.RandomUniform(0.1, 0.4, clock),
        pan=sf.RandomUniform(-1.0, 1.0, clock),
        rate=1.5,
    )
    for b in srcs
]
for g in granulator:
    g.set_buffer("envelope", sf.EnvelopeBuffer("linear-decay"))
granulator = sf.ChannelCrossfade(
    granulator, index=sf.RandomUniform(0, 2, clock=state), num_output_channels=1
)
granulator = sf.StereoPanner(granulator, 0.0)
granulator += sf.RandomImpulse(60.0) * 0.3

fb = sf.FeedbackBufferReader(fb_buf)
granulator += fb * 0.3

granulator = sf.Maximiser(granulator) * 0.3
fts = [sf.FFT(granulator[0]), sf.FFT(granulator[1])]
fts = [
    sf.FFTContinuousPhaseVocoder(fts[0], 0.5),
    sf.FFTContinuousPhaseVocoder(fts[1], 0.5),
]
fts = [sf.FFTContrast(fts[0], 0.9), sf.FFTContrast(fts[1], 0.9)]
fts = [sf.FFTTransform(fts[0], 0.0, -0.1), sf.FFTTransform(fts[1], 0.0, 0.1)]
fts = [sf.FFTTonality(fts[0], 0.2, 0.1), sf.FFTTonality(fts[1], 0.2, 0.1)]
fts = [sf.FFTNoiseGate(fts[0], 0.9), sf.FFTNoiseGate(fts[1], 0.9)]
fts = [sf.FFTConvolve(fts[0], kern), sf.FFTConvolve(fts[1], kern)]
ift = sf.ChannelArray([sf.IFFT(f) for f in fts])

graph.add_node(sf.FeedbackBufferWriter(fb_buf, ift, 1 / 60))

out = ift * sf.RandomBrownian(0.7, 0.9) + sf.SawOscillator(
    sf.RandomBrownian(59.9, 60.8)
) * sf.RandomBrownian(0.03, 0.05)

graph.add_node(sf.BufferRecorder(resample_buf, out))
out = sf.Granulator(
    resample_buf,
    clock=clock,
    rate=0.5,
    duration=sf.RandomUniform(0.1, 0.4, clock),
    pos=sf.RandomUniform(0.1, 0.9, state),
) * sf.SineOscillator(60.0)
out = sf.Maximiser(out)

graph.play(out)
graph.wait()
