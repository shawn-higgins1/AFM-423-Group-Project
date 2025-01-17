�	�W��P5@�W��P5@!�W��P5@	�e��ْ?�e��ْ?!�e��ْ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�W��P5@/�h��2@1�B=}��?A����z�?I�o}Xo� @Y�I���p?*	     �G@2U
Iterator::Model::ParallelMapV2S�!�uq�?!cr1��<@)S�!�uq�?1cr1��<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�
F%u�?!��
br;@)46<�R�?1�
br17@:Preprocessing2F
Iterator::ModelM�O��?!����F}E@)_�Q�{?1�Q�٨�,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateM�O��?!����F}5@) �o_�y?1�����*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!!&W�+ @)ŏ1w-!o?1!&W�+ @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipS�!�uq�?!cr1��L@)�~j�t�h?1&W�+�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!��F}g�@)��H�}]?1��F}g�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapA��ǘ��?!7�;��7@)����MbP?1��
br@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�e��ْ?ITFW�X@Q�T�^מ�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	/�h��2@/�h��2@!/�h��2@      ��!       "	�B=}��?�B=}��?!�B=}��?*      ��!       2	����z�?����z�?!����z�?:	�o}Xo� @�o}Xo� @!�o}Xo� @B      ��!       J	�I���p?�I���p?!�I���p?R      ��!       Z	�I���p?�I���p?!�I���p?b      ��!       JGPUY�e��ْ?b qTFW�X@y�T�^מ�?�"5
sequential/dense/MatMulMatMul�Nz�Ѷ�?!�Nz�Ѷ�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�~޼H�?!�f�H�e�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMuli�ZY�?!@����?"7
sequential/dense_1/MatMulMatMuli�ZY�?!��[_��?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchp�7�ڙ?!��P�y�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�~޼H�?!�����?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�����?!J*�?�?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam�Mm��?!+��b�8�?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam�Mm��?!��4�2�?"!
Adam/PowPow�Mm��?!�M$[�?Q      Y@Y��/Ċ�0@a�	�N]�T@q~�с�mW@y�M�v0�?"�
both�Your program is POTENTIALLY input-bound because 88.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�93.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 