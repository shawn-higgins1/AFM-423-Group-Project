�	�X�Ѝ5@�X�Ѝ5@!�X�Ѝ5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�X�Ѝ5@K:��l.3@1���^�2�?A������?I��w�� @*	gfffffI@2U
Iterator::Model::ParallelMapV2�Pk�w�?!���r�\;@)�Pk�w�?1���r�\;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatlxz�,C�?!K�R�T*;@)������?1�Z�V��6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate �o_Ή?!��l6��8@)��y�):�?1�P(
�1@:Preprocessing2F
Iterator::ModelM�O��?!9���C@) �o_�y?1��l6��(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vOn?!H$�D"@)���_vOn?1H$�D"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipK�=�U�?!���x<N@)a��+ei?1�F��h@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!U*�J�R@)/n��b?1U*�J�R@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�?�߾�?!�~����:@)/n��R?1U*�J�R@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIt�3��X@Q�E�w��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	K:��l.3@K:��l.3@!K:��l.3@      ��!       "	���^�2�?���^�2�?!���^�2�?*      ��!       2	������?������?!������?:	��w�� @��w�� @!��w�� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qt�3��X@y�E�w��?�"5
sequential/dense/MatMulMatMul]�-��4�?!]�-��4�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�(hB��?!?����?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad��2���?!V�7O;$�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��2���?!�W����?"7
sequential/dense_1/MatMulMatMul��2���?!�(hB��?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchk�P峿�?!����8��?")
sequential/CastCastm6O<~��?!��\�-�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�(hB��?!8�a����?"!
Adam/PowPow�(hB��?!��fFY�?"
Sum_7Sum��D[��?!⇯�>��?Q      Y@Y��/Ċ�0@a�	�N]�T@q��rX@y̲��!�?"�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 