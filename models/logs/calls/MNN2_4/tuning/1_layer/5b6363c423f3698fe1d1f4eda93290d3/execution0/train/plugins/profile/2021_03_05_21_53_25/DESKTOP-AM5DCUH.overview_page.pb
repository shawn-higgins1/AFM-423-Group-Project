�	"nN%�6@"nN%�6@!"nN%�6@	)l�Dl�?)l�Dl�?!)l�Dl�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6"nN%�6@�|A�4@1�27߈��?Ao�ŏ1�?If�L2r�?Y'.�+=�?*fffff�H@)       =2U
Iterator::Model::ParallelMapV2�!��u��?!���/��<@)�!��u��?1���/��<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��<,Ԋ?!��o�:@)��_vO�?1�c@�5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate������?!������7@)� �	�?1�q+�</@:Preprocessing2F
Iterator::ModelDio��ɔ?!������D@)a��+ey?1\����&)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!�:�Ǜ�@)�q����o?1�:�Ǜ�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipV-��?!jiiiiiM@)a��+ei?1\����&@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!
�-H�@)HP�s�b?1
�-H�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap-C��6�?!F=r4��9@)a2U0*�S?1��Ug�x@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 90.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9)l�Dl�?IR�a�X@Q�ɆO��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�|A�4@�|A�4@!�|A�4@      ��!       "	�27߈��?�27߈��?!�27߈��?*      ��!       2	o�ŏ1�?o�ŏ1�?!o�ŏ1�?:	f�L2r�?f�L2r�?!f�L2r�?B      ��!       J	'.�+=�?'.�+=�?!'.�+=�?R      ��!       Z	'.�+=�?'.�+=�?!'.�+=�?b      ��!       JGPUY)l�Dl�?b qR�a�X@y�ɆO��?�"5
sequential/dense/MatMulMatMulP�6�J�?!P�6�J�?0"C
%gradient_tape/sequential/dense/MatMulMatMulG`iV��?!L�ϫE��?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchY���?!b��V�w�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGradY���?!<���w�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulY���?!G`iV��?"7
sequential/dense_1/MatMulMatMulY���?!R��+&��?0"
Abs_1Absk��?!Y����?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdamk��?!`�,���?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdamk��?!gz7����?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdamk��?!n[Q,���?Q      Y@Y��/Ċ�0@a�	�N]�T@qo�̱FS@y�|CA�B�?"�
both�Your program is POTENTIALLY input-bound because 90.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�8.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�77.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 