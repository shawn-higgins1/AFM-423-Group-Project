�		kc섇5@	kc섇5@!	kc섇5@	?����*�??����*�?!?����*�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6	kc섇5@c���Y3@1��hW!��?AA��ǘ��?IÃf׽��?Y�>�p?*	     @H@2U
Iterator::Model::ParallelMapV2���QI�?!r���
|=@)���QI�?1r���
|=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeata��+e�?!i�n�'�9@)M�O��?1��,O"�4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate46<�R�?!���fy6@)_�Q�{?1|q���
,@:Preprocessing2F
Iterator::Model��_�L�?!���
|qE@)9��v��z?1�$2��*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	�^)�p?!��W?� @)	�^)�p?1��W?� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��e�c]�?!
|q���L@)Ǻ���f?1W?��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!���@)HP�s�b?1���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�������?!�:*��9@)-C��6Z?1���Id
@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?����*�?Ilp�é�X@Q���U��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	c���Y3@c���Y3@!c���Y3@      ��!       "	��hW!��?��hW!��?!��hW!��?*      ��!       2	A��ǘ��?A��ǘ��?!A��ǘ��?:	Ãf׽��?Ãf׽��?!Ãf׽��?B      ��!       J	�>�p?�>�p?!�>�p?R      ��!       Z	�>�p?�>�p?!�>�p?b      ��!       JGPUY?����*�?b qlp�é�X@y���U��?�"5
sequential/dense/MatMulMatMul}�p�R�?!}�p�R�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�ke+���?!���B�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulS}�n��?!"��0��?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch��>�l�?!��f~�?"!
Adam/PowPow�ke+���?!i�Q��?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�ke+���?!�D��H��?"7
sequential/dense_1/MatMulMatMul�ke+���?!i��۹D�?0"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam�!AO�?!�t	�N6�?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam�!AO�?!����'�?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam�!AO�?!�x,�x�?Q      Y@Y��/Ċ�0@a�	�N]�T@q�5~�#yW@y�f���H�?"�
both�Your program is POTENTIALLY input-bound because 89.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�93.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 