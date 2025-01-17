�	��Ia�W5@��Ia�W5@!��Ia�W5@	g#�f�s�?g#�f�s�?!g#�f�s�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��Ia�W5@��-�v�2@16�Ko.�?AM�St$�?I*��g\8@Y�l�/ڪ?*	43333�L@2U
Iterator::Model::ParallelMapV2U���N@�?!�s�U`@@)U���N@�?1�s�U`@@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���H�?!�{�J�;@)tF��_�?1��+Q�4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatǺ����?!��18�3@)HP�sׂ?1⾖�"0@:Preprocessing2F
Iterator::Model�{�Pk�?!�cOyF@)y�&1�|?1dp>�c(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!��t��@)����Mbp?1��t��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipvq�-�?!�R����K@)F%u�k?1$(ͦ�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!��t��@)����Mb`?1��t��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap������?!]h{�=@)a2U0*�S?1cOy�� @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9f#�f�s�?I��,g�X@Q��Oت�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��-�v�2@��-�v�2@!��-�v�2@      ��!       "	6�Ko.�?6�Ko.�?!6�Ko.�?*      ��!       2	M�St$�?M�St$�?!M�St$�?:	*��g\8@*��g\8@!*��g\8@B      ��!       J	�l�/ڪ?�l�/ڪ?!�l�/ڪ?R      ��!       Z	�l�/ڪ?�l�/ڪ?!�l�/ڪ?b      ��!       JGPUYf#�f�s�?b q��,g�X@y��Oت�?�"5
sequential/dense/MatMulMatMulj����?!j����?0"C
%gradient_tape/sequential/dense/MatMulMatMul%��q��?!�.ҐD��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��p`͡�?!�`����?"7
sequential/dense_1/MatMulMatMul��p`͡�?!`I���3�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch%��q��?!�������?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad%��q��?!*>��S�?"
Abs_1Abs����)?!�q�@�?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam����)?!���7,�?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam����)?!��N�Y�?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam����)?!V��P|�?Q      Y@Y��/Ċ�0@a�	�N]�T@q����0?W@yo�i���?"�
both�Your program is POTENTIALLY input-bound because 88.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�93.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 