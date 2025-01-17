�	,���h7@,���h7@!,���h7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-,���h7@0��4@1���)�?A|�Pk��?I�!8.�f@*	33333�L@2U
Iterator::Model::ParallelMapV2	�^)ː?!�]dS<@)	�^)ː?1�]dS<@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate;�O��n�?!Q�ݙ�?@)lxz�,C�?1�n��7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatg��j+��?!(b6�54@)��ׁsF�?1ӡ�31@:Preprocessing2F
Iterator::Model0*��D�?!�\:�`wD@)�<,Ԛ�}?1��-�D7)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�J�4q?!­���@)�J�4q?1­���@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip������?!&��+��M@)a��+ei?1eI�^�j@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!�~���@)��H�}]?1�~���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�j+��ݓ?!�V��@@)Ǻ���V?1�s~v�W@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�5;��X@Q���2���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	0��4@0��4@!0��4@      ��!       "	���)�?���)�?!���)�?*      ��!       2	|�Pk��?|�Pk��?!|�Pk��?:	�!8.�f@�!8.�f@!�!8.�f@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�5;��X@y���2���?�"5
sequential/dense/MatMulMatMult�v��K�?!t�v��K�?0"C
%gradient_tape/sequential/dense/MatMulMatMul5�F�@�?!>�(PF�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul5�F�@�?!��>��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul5�F�@�?!#1�C�?"7
sequential/dense_1/MatMulMatMul
�Z2�?!s��q�	�?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul==�u+�?!6��F�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchH��萕?!dv��2��?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradH��萕?!���P��?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam5�F�@�?!�گ3i�?"7
sequential/dense_2/MatMulMatMul5�F�@�?!��GT?}�?0Q      Y@Y>����/@aX�i��U@qs� �iX@y��x��?"�
both�Your program is POTENTIALLY input-bound because 88.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 