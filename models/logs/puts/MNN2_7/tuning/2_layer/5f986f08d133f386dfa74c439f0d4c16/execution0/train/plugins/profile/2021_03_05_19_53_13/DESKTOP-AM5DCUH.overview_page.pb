�	����9@����9@!����9@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-����9@�.5B?�6@1��DK�?A�^)�Ǫ?Ih׿�@*	43333sH@2U
Iterator::Model::ParallelMapV2-C��6�?!A�9�-:@)-C��6�?1A�9�-:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatlxz�,C�?!�f5�8<@)Zd;�O��?1��,s�7@:Preprocessing2F
Iterator::Model{�G�z�?!b��,sD@)��H�}}?1ɀz�r-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate{�G�z�?!b��,s4@)F%u�{?1�bK�m�*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice_�Q�k?!�/]��@)_�Q�k?1�/]��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��j+���?!�C�ӌM@)�~j�t�h?1��f5�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!�c�#\�@)HP�s�b?1�c�#\�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��@��ǈ?!	p�q��8@)�J�4a?1��uǋ-@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIWE�̊X@QY�n��L�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�.5B?�6@�.5B?�6@!�.5B?�6@      ��!       "	��DK�?��DK�?!��DK�?*      ��!       2	�^)�Ǫ?�^)�Ǫ?!�^)�Ǫ?:	h׿�@h׿�@!h׿�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qWE�̊X@yY�n��L�?�"C
%gradient_tape/sequential/dense/MatMulMatMul�;��?!�;��?0"5
sequential/dense/MatMulMatMul�;��?!�;��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul����R�?!~���r�?"E
'gradient_tape/sequential/dense_1/MatMulMatMul��ej�Q�?!3x0ځ��?0"7
sequential/dense_1/MatMulMatMul��ej�Q�?!'0}'���?0"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdamė�
��?! #�w �?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchė�
��?!�yr�?"7
sequential/dense_2/MatMulMatMul2ظs�?!Y�����?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMulE̪-gT�?!���}K�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam��ej�Q�?!7L �?Q      Y@Y>����/@aX�i��U@q��E�tX@y�}#�w�?"�
both�Your program is POTENTIALLY input-bound because 88.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 