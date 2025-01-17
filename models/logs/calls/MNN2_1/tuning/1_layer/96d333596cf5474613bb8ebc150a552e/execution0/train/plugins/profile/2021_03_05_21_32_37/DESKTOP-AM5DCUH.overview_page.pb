�	�|$%=l6@�|$%=l6@!�|$%=l6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�|$%=l6@�O�m�3@1̘�5Φ�?A�I+��?I���I'@*	43333sH@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate;�O��n�?!�}��gB@)L7�A`�?1�/��@@:Preprocessing2U
Iterator::Model::ParallelMapV2���S㥋?!x��|�;@)���S㥋?1x��|�;@:Preprocessing2F
Iterator::Model{�G�z�?!b��,sD@)9��v��z?1�|Bٹ�*@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatS�!�uq{?!KIT"g+@)/n��r?1L�����!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��j+���?!�C�ӌM@)�~j�t�h?1��f5�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!�c�#\�@)HP�s�b?1�c�#\�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa2U0*��?!�0�QġC@)a2U0*�S?1�0�Qġ@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��H�}M?!ɀz�r�?)��H�}M?1ɀz�r�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!�0�Qġ�?)a2U0*�C?1�0�Qġ�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIq3�ѬX@Q�#sz���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�O�m�3@�O�m�3@!�O�m�3@      ��!       "	̘�5Φ�?̘�5Φ�?!̘�5Φ�?*      ��!       2	�I+��?�I+��?!�I+��?:	���I'@���I'@!���I'@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qq3�ѬX@y�#sz���?�"5
sequential/dense/MatMulMatMul=�T��%�?!=�T��%�?0"C
%gradient_tape/sequential/dense/MatMulMatMul1,w�/�?!������?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��Q��&�?!c}:-��?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch=�T��%�?!��g���?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad=�T��%�?!�|2�I��?"7
sequential/dense_1/MatMulMatMul=�T��%�?!i�� ��?0"]
?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nanDivNoNan/1,w�/�?!|�o����?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdamJ�2�G-�?!�
�0ҩ�?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdamJ�2�G-�?!�6�����?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdamJ�2�G-�?!�b�&{o�?Q      Y@Y{	�%��1@a�����T@q��wZ�mX@y5�*o�?"�
both�Your program is POTENTIALLY input-bound because 87.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 