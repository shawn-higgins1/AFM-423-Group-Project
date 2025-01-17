�	QlMK�5@QlMK�5@!QlMK�5@	�� A�?�� A�?!�� A�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6QlMK�5@���o{�2@1
H�`��?AEGr��?I������@Y�tu�b�t?*	������J@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatX9��v��?!�/Ċ��<@)F%u��?1����B�8@:Preprocessing2U
Iterator::Model::ParallelMapV2-C��6�?!�Ե��7@)-C��6�?1�Ե��7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�?�߾�?!靺���9@)��ׁsF�?1��4>2�2@:Preprocessing2F
Iterator::Model��~j�t�?!	�N]��A@)a��+ey?1H&�;u-'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipD�l����?!��XQP@)����Mbp?1*J�#�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!��bEi@)ŏ1w-!o?1��bEi@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!��4>2@)HP�s�b?1��4>2@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���Q��?!���	<@)��_�LU?1�S�rp@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 86.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�� A�?IW�Y[��X@Q�P,ѱ��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���o{�2@���o{�2@!���o{�2@      ��!       "	
H�`��?
H�`��?!
H�`��?*      ��!       2	EGr��?EGr��?!EGr��?:	������@������@!������@B      ��!       J	�tu�b�t?�tu�b�t?!�tu�b�t?R      ��!       Z	�tu�b�t?�tu�b�t?!�tu�b�t?b      ��!       JGPUY�� A�?b qW�Y[��X@y�P,ѱ��?�"5
sequential/dense/MatMulMatMul	���磦?!	���磦?0"C
%gradient_tape/sequential/dense/MatMulMatMul����?!�#�G�a�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul
f�E�'�?!
=_�⫻?"7
sequential/dense_1/MatMulMatMul
f�E�'�?!F+c����?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch����?!G&I�~�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad����?!H����?"!
Adam/PowPow(]��jϓ?!��\��|�?"O
5gradient_tape/mean_squared_error/weighted_loss/Tile_1Tile���s�V�?!���i���?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam"��/�?!� �h���?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam"��/�?!
"~g���?Q      Y@Y��/Ċ�0@a�	�N]�T@q�<%�kX@y���q�?"�
both�Your program is POTENTIALLY input-bound because 86.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 