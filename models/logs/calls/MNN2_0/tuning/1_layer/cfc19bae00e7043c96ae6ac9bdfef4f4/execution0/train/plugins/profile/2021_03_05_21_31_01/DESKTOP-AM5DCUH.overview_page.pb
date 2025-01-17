�	+hZbe�5@+hZbe�5@!+hZbe�5@	��G�&�?��G�&�?!��G�&�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6+hZbe�5@TT�J�[3@1���f�?Ayv�և�?I�^D�1 @Y�ʅʿ�w?*	�����YI@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�l����?!�MR��>B@)X�5�;N�?1������@@:Preprocessing2U
Iterator::Model::ParallelMapV2?�ܵ�|�?!>��xc�?@)?�ܵ�|�?1>��xc�?@:Preprocessing2F
Iterator::ModelA��ǘ��?!���ջ�E@)�HP�x?1*w�d((@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat �o_�y?!�HPS!�(@)HP�s�r?1�Ӈi]%"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�v��/�?!	j*DL@)��_�Le?14H�4H�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor_�Q�[?!��!��
@)_�Q�[?1��!��
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapn���?!��RTC@)/n��R?14{d[@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor����MbP?!۽=��?)����MbP?1۽=��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!��WV��?)a2U0*�C?1��WV��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��G�&�?I��H��X@QMnh��T�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	TT�J�[3@TT�J�[3@!TT�J�[3@      ��!       "	���f�?���f�?!���f�?*      ��!       2	yv�և�?yv�և�?!yv�և�?:	�^D�1 @�^D�1 @!�^D�1 @B      ��!       J	�ʅʿ�w?�ʅʿ�w?!�ʅʿ�w?R      ��!       Z	�ʅʿ�w?�ʅʿ�w?!�ʅʿ�w?b      ��!       JGPUY��G�&�?b q��H��X@yMnh��T�?�"5
sequential/dense/MatMulMatMulpȄ��	�?!pȄ��	�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�-�~z�?!������?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch@yD���?!n"V(�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul@yD���?!ߛ�G�?"7
sequential/dense_1/MatMulMatMul@yD���?!+�r+z�?0"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�-�~z�?!��u�k	�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�-�~z�?!{vI���?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam�����?!�2(N��?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam�����?!���o�?"!
Adam/PowPow�����?!\����Z�?Q      Y@Y{	�%��1@a�����T@q��?�gW@y������?"�
both�Your program is POTENTIALLY input-bound because 89.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�93.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 