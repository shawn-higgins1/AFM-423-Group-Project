�	���T�7@���T�7@!���T�7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���T�7@/5B?S�4@1r�_!s�?A_)�Ǻ�?I4��M@*	������H@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate��A�f�?!2�c�E@)�j+��ݓ?1�9�s�C@:Preprocessing2U
Iterator::Model::ParallelMapV2 �o_Ή?!�s�9g9@) �o_Ή?1�s�9g9@:Preprocessing2F
Iterator::Model�:pΈ�?!�{��>B@)�I+�v?1���Zk-&@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeata��+ey?!      )@)"��u��q?1�Zk��V!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipT㥛� �?!"�B�O@)�����g?1c�1�c@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!��RJ)�@)ŏ1w-!_?1��RJ)�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapǺ���?!�RJ)��F@)�~j�t�X?12�c�1@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor-C��6J?!�9�s��?)-C��6J?1�9�s��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!�RJ)���?)Ǻ���F?1�RJ)���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI���8�X@QUO����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	/5B?S�4@/5B?S�4@!/5B?S�4@      ��!       "	r�_!s�?r�_!s�?!r�_!s�?*      ��!       2	_)�Ǻ�?_)�Ǻ�?!_)�Ǻ�?:	4��M@4��M@!4��M@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q���8�X@yUO����?�"5
sequential/dense/MatMulMatMul���V�?!���V�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�,���?!T�?Za4�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�,���?!�-p��?0"7
sequential/dense_1/MatMulMatMul�׬󰫠?!E�nS�	�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulKθ�ߝ?!ԥϚ��?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul���V�?!�qop�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�,���?!��5&���?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�,���?!<������?"7
sequential/dense_2/MatMulMatMul�,���?!�[��:��?0"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam�׬󰫐?!e{WXv�?Q      Y@Y�C=�C=0@a��
��T@q��0=�xX@y�%�RN��?"�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 