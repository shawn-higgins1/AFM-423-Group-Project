�	�F�g�m7@�F�g�m7@!�F�g�m7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�F�g�m7@��t�(�4@1���@��?Aђ����?IгY���@*	43333�F@2U
Iterator::Model::ParallelMapV2�+e�X�?!�+ր+�8@)�+e�X�?1�+ր+�8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�
F%u�?!�Ur9�;@)/�$��?1�5��5�6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��_vO�?!������7@)S�!�uq{?1���j�1-@:Preprocessing2F
Iterator::Model��y�):�?!�*���cC@)-C��6z?1�T8q��+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	�^)�p?!P��k�!@)	�^)�p?1P��k�!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�?Ɯ?!�Fc*�N@)_�Q�k?1�Kh/�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!H�֝,@)/n��b?1H�֝,@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�~j�t��?!{�$z�$:@)a2U0*�S?1�?�?�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI54��X@Q��2����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��t�(�4@��t�(�4@!��t�(�4@      ��!       "	���@��?���@��?!���@��?*      ��!       2	ђ����?ђ����?!ђ����?:	гY���@гY���@!гY���@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q54��X@y��2����?�"5
sequential/dense/MatMulMatMulO����?!O����?0"C
%gradient_tape/sequential/dense/MatMulMatMul��X�?!�~�m�?0"7
sequential/dense_1/MatMulMatMul�_LX�?!���7��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul 1�<[�?!oܸ�[X�?"E
'gradient_tape/sequential/dense_1/MatMulMatMul%k�'Z�?!��m��#�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch��w���?!��\	d��?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul��w���?!@�K�'��?"7
sequential/dense_2/MatMulMatMul��w���?!��:�D�?0"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam�_LX�?!T����o�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�_LX�?!YF)Jx��?Q      Y@Y>����/@aX�i��U@q�����X@yR�!:S�?"�
both�Your program is POTENTIALLY input-bound because 89.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 